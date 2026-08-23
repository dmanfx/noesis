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

    def test_missing_pgie_path_uses_safe_bbox_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime_config.apply_osd_from_pgie_ini({}, Path(tmp) / "infer.yaml")
            self.assertEqual(result["osd"]["display-mask"], 0)
            self.assertEqual(result["osd"]["display-bbox"], 1)
            self.assertEqual(result["osd"]["process-mode"], 1)


if __name__ == "__main__":
    unittest.main()
