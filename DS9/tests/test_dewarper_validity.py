from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from geometry.dewarper_validity import build_dewarper_fov_mask, load_dewarper_fov_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_CONFIG = REPO_ROOT / "DS9" / "config" / "infer.yaml"


class Ds9DewarperValidityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(DS9_CONFIG.read_text(encoding="utf-8"))

    def test_all_ds9_sources_declare_calibrated_circular_validity(self) -> None:
        masks = self.config["dewarper_validity_masks"]["sources"]
        self.assertEqual(set(masks), {"0", "1", "2"})
        for source_id, expected_size in (("0", (1920, 1080)), ("1", (1920, 1080)), ("2", (1280, 720))):
            mask = masks[source_id]
            self.assertEqual(mask["source-valid-region"], "circle")
            self.assertEqual((mask["source-width"], mask["source-height"]), expected_size)
            self.assertEqual(mask["erode-px"], 1)

    def test_family_room_mask_uses_ds9_dewarper_and_excludes_border(self) -> None:
        source_cfg = self.config["sources"][2]
        mask_cfg = self.config["dewarper_validity_masks"]["sources"]["2"]
        spec = load_dewarper_fov_spec(
            source_cfg=source_cfg,
            pipeline_yaml_path=DS9_CONFIG,
            mask_cfg=mask_cfg,
            repo_root=REPO_ROOT,
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.config_path, REPO_ROOT / "DS9" / "config" / "dewarper_g4_instant_charuco_720_to_1080.txt")
        self.assertEqual(spec.source_size, (1280, 720))
        mask = build_dewarper_fov_mask(spec, target_size=(1920, 1080), erode_px=1)
        self.assertEqual(mask.shape, (1080, 1920))
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertGreater(float(mask.mean()), 0.75)
        self.assertLess(float(mask.mean()), 0.85)
        self.assertEqual(int(mask[:, -1].sum()), 0)

    def test_ds9_mapanything_preserves_dense_depth_and_downweights_uncertainty(self) -> None:
        hooks = (REPO_ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py").read_text(encoding="utf-8")
        self.assertIn("self._dewarper_validity_mask(source_id", hooks)
        self.assertIn("strict_mask = finite_depth & model_mask", hooks)
        self.assertIn("mask = finite_depth", hooks)
        self.assertIn("depth[~mask] = np.nan", hooks)
        self.assertIn("_MAPANYTHING_AMBIGUOUS_CONFIDENCE_SCALE", hooks)
        self.assertIn("_MAPANYTHING_OUTSIDE_CALIBRATED_FOV_CONFIDENCE_SCALE", hooks)
        self.assertNotIn("Depth all-NaN/invalid; filled zeros to emit frame", hooks)


if __name__ == "__main__":
    unittest.main()
