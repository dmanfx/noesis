from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_runtime_config():
    path = REPO_ROOT / "DS9" / "noesis" / "runtime_config.py"
    spec = importlib.util.spec_from_file_location("ds9_runtime_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime_config = _load_runtime_config()


class RuntimeConfigTests(unittest.TestCase):
    def test_mask_pgie_enables_masks_and_preserves_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "pgie.ini"
            ini.write_text(
                "[property]\nnetwork-type=3\noutput-instance-mask=1\n"
                "parse-bbox-instance-mask-func-name=ParseMasks\n",
                encoding="utf-8",
            )
            result = runtime_config.apply_osd_from_pgie_ini(
                {"models": {"pgie": {"config-file-path": str(ini)}}},
                root / "infer.yaml",
            )
            self.assertEqual(result["osd"]["display-mask"], 1)
            self.assertEqual(result["osd"]["display-bbox"], 1)
            self.assertEqual(result["osd"]["process-mode"], 1)

    def test_detect_only_pgie_disables_masks_and_shows_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "pgie.ini"
            ini.write_text("[property]\nnetwork-type=0\noutput-instance-mask=0\n", encoding="utf-8")
            result = runtime_config.apply_osd_from_pgie_ini(
                {"models": {"pgie": {"config-file-path": str(ini)}}},
                root / "infer.yaml",
            )
            self.assertEqual(result["osd"]["display-mask"], 0)
            self.assertEqual(result["osd"]["display-bbox"], 1)
            self.assertEqual(result["osd"]["process-mode"], 1)

    def test_v3dt_preserves_bbox_off_for_detector_pgie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "detector.ini"
            ini.write_text(
                "[property]\nnetwork-type=0\noutput-instance-mask=0\n",
                encoding="utf-8",
            )
            result = runtime_config.apply_osd_from_pgie_ini(
                {
                    "tracking_mode": "v3dt",
                    "osd": {
                        "process-mode": 0,
                        "display-mask": 0,
                        "display-bbox": 0,
                        "display-text": 0,
                    },
                    "models": {"pgie": {"config-file-path": str(ini)}},
                },
                root / "infer.yaml",
            )
            self.assertEqual(
                result["osd"],
                {
                    "process-mode": 1,
                    "display-mask": 0,
                    "display-bbox": 0,
                    "display-text": 1,
                },
            )

    def test_v3dt_seg_pgie_keeps_mask_and_gpu_capabilities_with_bbox_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "seg_pgie.ini"
            ini.write_text(
                "[property]\nnetwork-type=3\noutput-instance-mask=1\n"
                "parse-bbox-instance-mask-func-name=ParseMasks\n",
                encoding="utf-8",
            )
            result = runtime_config.apply_osd_from_pgie_ini(
                {
                    "tracking_mode": "v3dt",
                    "osd": {"display-mask": 0, "display-bbox": 0, "display-text": 0},
                    "models": {"pgie": {"config-file-path": str(ini)}},
                },
                root / "infer.yaml",
            )
            self.assertEqual(result["osd"]["process-mode"], 1)
            self.assertEqual(result["osd"]["display-mask"], 1)
            self.assertEqual(result["osd"]["display-bbox"], 0)
            self.assertEqual(result["osd"]["display-text"], 1)

    def test_v3dt_bbox_on_remains_available_for_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "detector.ini"
            ini.write_text("[property]\nnetwork-type=0\n", encoding="utf-8")
            result = runtime_config.apply_osd_from_pgie_ini(
                {
                    "tracking_mode": "v3dt",
                    "osd": {"display-bbox": 1},
                    "models": {"pgie": {"config-file-path": str(ini)}},
                },
                root / "infer.yaml",
            )
            self.assertEqual(result["osd"]["display-bbox"], 1)

    def test_non_v3dt_modes_keep_derived_bbox_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = root / "detector.ini"
            ini.write_text(
                "[property]\nnetwork-type=0\noutput-instance-mask=0\n",
                encoding="utf-8",
            )
            for tracking_mode in ("baseline", "mv3dt"):
                with self.subTest(tracking_mode=tracking_mode):
                    result = runtime_config.apply_osd_from_pgie_ini(
                        {
                            "tracking_mode": tracking_mode,
                            "osd": {"display-bbox": 0},
                            "models": {"pgie": {"config-file-path": str(ini)}},
                        },
                        root / "infer.yaml",
                    )
                    self.assertEqual(result["osd"]["display-mask"], 0)
                    self.assertEqual(result["osd"]["display-bbox"], 1)
                    self.assertEqual(result["osd"]["process-mode"], 1)

    def test_missing_pgie_path_uses_safe_bbox_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime_config.apply_osd_from_pgie_ini({}, Path(tmp) / "infer.yaml")
            self.assertEqual(result["osd"]["display-mask"], 0)
            self.assertEqual(result["osd"]["display-bbox"], 1)
            self.assertEqual(result["osd"]["process-mode"], 1)


if __name__ == "__main__":
    unittest.main()
