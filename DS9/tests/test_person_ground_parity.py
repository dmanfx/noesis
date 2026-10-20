from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


@lru_cache(maxsize=1)
def _characterize(adapter_root: Path) -> dict[str, object]:
    script = r'''
import json
import sys
from pathlib import Path

import numpy as np

adapter = Path(sys.argv[1]).resolve()
repo = Path(sys.argv[2]).resolve()
sys.path = [str(adapter), str(repo)] + [
    value for value in sys.path
    if value and Path(value).resolve() not in {adapter, repo}
]

from noesis.pipelines import hooks
from noesis.telemetry.person_ground_state import HumanGroundConfig

processor = object.__new__(hooks._AnalyticsTelemetryProcessor)
processor._human_ground_cfg = HumanGroundConfig(max_speed_mps=2.0, max_jump_m=0.5)

keypoints = np.zeros((17, 3), dtype=np.float32)
keypoints[5] = [100.0, 120.0, 0.9]
keypoints[6] = [140.0, 120.0, 0.9]
keypoints[11] = [110.0, 190.0, 0.9]
keypoints[12] = [130.0, 190.0, 0.9]
keypoints[13] = [150.0, 210.0, 0.9]
keypoints[14] = [160.0, 210.0, 0.9]
keypoints[15] = [155.0, 215.0, 0.5]
keypoints[16] = [165.0, 215.0, 0.5]
candidate = processor._resolve_pose_floor_anchor(keypoints, posture="sitting")

state = hooks._WorldAnchorState()
first = processor._update_world_state(
    state,
    measurement=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=0.0,
    alpha=0.5,
    beta=0.1,
    quality="good",
)
state.motion_mode = "walk"
second = processor._update_world_state(
    state,
    measurement=np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=0.1,
    alpha=0.5,
    beta=0.1,
    quality="good",
)

world_key = processor._world_track_key(
    4,
    {"stable_id": 9, "tracker_id": 77, "track_id": 88},
)

long_gap_state = hooks._WorldAnchorState()
processor._update_world_state(
    long_gap_state,
    measurement=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=0.0,
    alpha=0.5,
    beta=0.1,
    quality="good",
)
long_gap_output = processor._update_world_state(
    long_gap_state,
    measurement=np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=2.0,
    alpha=0.5,
    beta=0.1,
    quality="good",
)

reacquire_state = hooks._WorldAnchorState()
processor._update_world_state(
    reacquire_state,
    measurement=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=0.0,
    alpha=0.5,
    beta=0.1,
    quality="good",
)
reacquire_outputs = []
reacquire_diagnostics = []
for x, ts in ((10.0, 2.0), (10.1, 2.1), (10.2, 2.2)):
    output = processor._update_world_state(
        reacquire_state,
        measurement=np.asarray([x, 0.0, 0.0], dtype=np.float64),
        floor_y=0.0,
        now_ts=ts,
        alpha=0.5,
        beta=0.1,
        quality="good",
    )
    reacquire_outputs.append([round(float(value), 9) for value in output])
    reacquire_diagnostics.append(dict(reacquire_state.as_public_fields()))

prediction_state = hooks._WorldAnchorState()
processor._update_world_state(
    prediction_state,
    measurement=np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
    floor_y=0.0,
    now_ts=1.0,
    alpha=0.5,
    beta=0.1,
    quality="good",
)
prediction_state.vel_world_x = 0.5
prediction_state.vel_world_z = -0.25
prediction_track = {}
prediction_output = processor._update_track_world_state(
    prediction_track,
    prediction_state,
    measurement=np.asarray([1.1, 0.0, 1.9], dtype=np.float64),
    floor_y=0.0,
    now_ts=1.2,
    alpha=0.5,
    beta=0.1,
    quality="good",
)

print(json.dumps({
    "candidate_source": candidate.source if candidate is not None else None,
    "state_owner": hooks._WorldAnchorState.__module__,
    "first": [round(float(value), 9) for value in first],
    "second": [round(float(value), 9) for value in second],
    "world_key": list(world_key) if world_key is not None else None,
    "long_gap": {
        "output": [round(float(value), 9) for value in long_gap_output],
        "diagnostics": dict(long_gap_state.as_public_fields()),
    },
    "reacquire": {
        "outputs": reacquire_outputs,
        "diagnostics": reacquire_diagnostics,
    },
    "prediction": {
        "prior": prediction_track.get("world_filter_prediction"),
        "measurement": prediction_track.get("world_prefilter_measurement"),
        "output": [round(float(value), 9) for value in prediction_output],
    },
    "public_keys": sorted(reacquire_state.as_public_fields()),
}, sort_keys=True))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(adapter_root), str(REPO_ROOT)],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


class PersonGroundContractTests(unittest.TestCase):
    def test_registered_body_support_uses_observed_not_held_posture(self) -> None:
        tree = ast.parse((DS9_ROOT / "noesis" / "pipelines" / "hooks.py").read_text())
        operator = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "_AnalyticsTelemetryProcessor")
        augment = next(node for node in operator.body if isinstance(node, ast.FunctionDef)
                       and node.name == "_augment_track_with_world")
        calls = [node for node in ast.walk(augment) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "_append_universal_world_candidates"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
        self.assertEqual(keywords["posture"], "str(posture)")
        self.assertEqual(keywords["observed_posture"], "str(raw_classified_posture)")
        # The independent exact-original torso branch also receives OBSERVED
        # posture; it must not turn the held safety classification into evidence.
        strong_call, = [node for node in ast.walk(augment) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_append_strong_posed_column"]
        self.assertEqual({k.arg: ast.unparse(k.value) for k in strong_call.keywords}["posture"],
                         "raw_classified_posture")
        strong = next(node for node in operator.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_append_strong_posed_column")
        column_call, = [node for node in ast.walk(strong) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "_append_universal_world_candidates"]
        column_keywords = {k.arg: ast.unparse(k.value) for k in column_call.keywords}
        self.assertEqual(column_keywords["posture"], "posture")
        self.assertEqual(column_keywords["observed_posture"], "posture")
        self.assertEqual(column_keywords["registered_column_only"], "True")
        self.assertEqual(column_keywords["depth_result"], "proof.depth")
        assignments = {node.targets[0].id: ast.unparse(node.value)
                       for node in ast.walk(augment) if isinstance(node, ast.Assign)
                       and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in {"exact_current_non_upright_torso",
                                                  "torso_context_supported"}}
        self.assertIn("raw_classified_posture", assignments["exact_current_non_upright_torso"])
        self.assertIn("raw_classified_posture", assignments["torso_context_supported"])
        self.assertIn("occlusion_assessment.active", assignments["torso_context_supported"])
        self.assertIn("partial_body_without_floor_contact", assignments["torso_context_supported"])

    def test_observed_posture_caller_is_separate_from_held_seat_policy(self) -> None:
        """Check the executable caller, not a second posture implementation."""
        path = DS9_ROOT / "noesis" / "pipelines" / "hooks.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "assess_lower_body_occlusion"]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: ast.unparse(item.value) for item in calls[0].keywords}
        self.assertEqual(keywords["posture"], "posture")
        self.assertEqual(keywords["observed_posture"], "raw_classified_posture")
        guards = [ast.unparse(node.test) for node in ast.walk(tree)
                  if isinstance(node, ast.If)
                  and any(isinstance(item, ast.Call)
                          and isinstance(item.func, ast.Attribute)
                          and item.func.attr == "_seated_torso_height_from_registered_depth"
                          for item in ast.walk(node))
                  and "posture == 'sitting'" in ast.unparse(node.test)]
        self.assertEqual(len(guards), 1)
        self.assertIn("raw_classified_posture == 'sitting'", guards[0])

    def test_ds9_adapter_uses_shared_product_state_without_local_clone(self) -> None:
        path = DS9_ROOT / "noesis" / "pipelines" / "hooks.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        local_classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertNotIn("_WorldAnchorState", local_classes)
        self.assertNotIn("_PoseAnchorCandidate", local_classes)
        self.assertIn("def _predict_world_state", text)
        self.assertIn("from noesis.telemetry.person_ground_state import", text)
        self.assertIn("classify_posture(", text)
        self.assertIn("assess_lower_body_occlusion(", text)
        self.assertIn("begin_source_admission(", text)
        self.assertIn("complete_source_admission(", text)
        self.assertIn("observe_coherent_image_motion(", text)
        self.assertNotIn("apply_source_hysteresis(", text)
        self.assertIn("update_motion_mode(", text)

    def test_ds91_ground_adapter_characterization_matches_product_contract(self) -> None:
        ds9 = _characterize(DS9_ROOT)
        self.assertEqual(ds9["candidate_source"], "pose_ankle_floor")
        self.assertEqual(ds9["state_owner"], "noesis.telemetry.person_ground_state")
        self.assertEqual(ds9["prediction"]["prior"], [1.1, 0.0, 1.95])

    def test_world_key_prefers_tracker_lifecycle_over_stable_identity(self) -> None:
        ds9 = _characterize(DS9_ROOT)
        self.assertEqual(ds9["world_key"], [4, 77])

    def test_long_gap_impossible_measurement_is_rejected(self) -> None:
        result = _characterize(DS9_ROOT)["long_gap"]
        self.assertEqual(result["output"], [0.0, 0.0, 0.0])
        diagnostics = result["diagnostics"]
        self.assertFalse(diagnostics["world_measurement_accepted"])
        self.assertEqual(
            diagnostics["world_rejection_reason"],
            "physical_innovation_exceeded",
        )
        self.assertEqual(diagnostics["world_reacquire_count"], 0)
        self.assertFalse(diagnostics["world_reacquired"])
        self.assertFalse(diagnostics["trail_break_required"])

    def test_consistent_impossible_measurements_without_image_motion_do_not_reacquire(self) -> None:
        result = _characterize(DS9_ROOT)["reacquire"]
        self.assertEqual(
            result["outputs"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        first, second, third = result["diagnostics"]
        self.assertEqual(first["world_reacquire_count"], 0)
        self.assertEqual(second["world_reacquire_count"], 0)
        self.assertEqual(third["world_reacquire_count"], 0)
        self.assertFalse(first["world_measurement_accepted"])
        self.assertFalse(second["world_measurement_accepted"])
        self.assertFalse(third["world_measurement_accepted"])
        self.assertFalse(third["world_reacquired"])
        self.assertFalse(third["trail_break_required"])

    def test_public_filter_diagnostic_keys_are_exposed(self) -> None:
        public_keys = set(_characterize(DS9_ROOT)["public_keys"])
        self.assertTrue(
            {
                "world_measurement_accepted",
                "world_rejection_reason",
                "world_innovation_m",
                "world_innovation_limit_m",
                "world_reacquire_count",
                "world_reacquired",
                "world_contact_basis",
                "world_image_motion_supported",
                "world_image_motion_streak",
                "trail_break_required",
                "lower_body_occluded",
                "lower_body_occlusion_level",
                "lower_body_occlusion_confidence",
            }.issubset(public_keys)
        )

    def test_ds9_bev_consumes_ground_fields_and_shared_path_commit(self) -> None:
        self.assertFalse(
            (DS9_ROOT / "noesis" / "telemetry" / "bev.py").exists()
        )
        path = REPO_ROOT / "noesis" / "telemetry" / "bev.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn("HumanGroundConfig", text)
        self.assertIn("commit_path_point(", text)
        for field in (
            "motion_mode",
            "posture",
            "trail_append_allowed",
            "idle_jitter_m",
        ):
            self.assertIn(field, text)


if __name__ == "__main__":
    unittest.main()
