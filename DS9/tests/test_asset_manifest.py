from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_validator():
    path = REPO_ROOT / "DS9" / "scripts" / "validate_asset_manifest.py"
    spec = importlib.util.spec_from_file_location("validate_asset_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


class AssetManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = yaml.safe_load((REPO_ROOT / "DS9" / "asset_manifest.yaml").read_text(encoding="utf-8"))

    def test_manifest_structure_and_declared_states_match_disk(self) -> None:
        result = validator.validate_manifest(self.manifest)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["artifact_count"], len(self.manifest["artifacts"]))
        self.assertEqual(
            sum(result["state_counts"].values()), result["artifact_count"]
        )

    def test_runtime_image_authority_is_separate_and_layered_on_build_image(self) -> None:
        build_image = self.manifest["target"]["build_image"]
        runtime_image = self.manifest["runtime"]["image"]

        self.assertEqual(runtime_image, validator.RUNTIME_IMAGE_AUTHORITY)
        self.assertEqual(runtime_image["parent_reference"], build_image["reference"])
        self.assertEqual(runtime_image["parent_image_id"], build_image["image_id"])
        self.assertNotEqual(runtime_image["reference"], build_image["reference"])
        self.assertNotEqual(runtime_image["image_id"], build_image["image_id"])
        dockerfile = REPO_ROOT / runtime_image["dockerfile"]
        self.assertEqual(validator._hash_file(dockerfile), runtime_image["dockerfile_sha256"])

    def test_runtime_image_authority_drift_is_rejected(self) -> None:
        for key, value in (
            ("image_id", "sha256:" + "0" * 64),
            ("parent_image_id", "sha256:" + "1" * 64),
            ("dockerfile_sha256", "2" * 64),
        ):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["runtime"]["image"][key] = value
                result = validator.validate_manifest(manifest)
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("runtime.image" in error for error in result["errors"]),
                    result["errors"],
                )

    def test_canonical_file_gate_reports_missing_artifacts(self) -> None:
        result = validator.validate_manifest(self.manifest, check_files=True, profile="canonical")
        self.assertFalse(result["ok"])
        self.assertTrue(any("engine.mapanything" in blocker for blocker in result["blockers"]))
        self.assertTrue(any("engine.yolo26_detect_m" in blocker for blocker in result["blockers"]))
        self.assertFalse(any("engine.yolo11_seg_alternate" in blocker for blocker in result["blockers"]))
        self.assertFalse(any("plugin.roi_exclude" in blocker for blocker in result["blockers"]))
        self.assertFalse(any("plugin.force_idr" in blocker for blocker in result["blockers"]))
        self.assertFalse(any("plugin.orderly_eos" in blocker for blocker in result["blockers"]))

    def test_provenance_gate_accepts_cpu_binaries_and_rejects_unbuilt_engines(self) -> None:
        result = validator.validate_manifest(self.manifest, profile="canonical", require_provenance=True)
        self.assertFalse(result["ok"])
        self.assertTrue(any("engine.mapanything" in error for error in result["errors"]))
        self.assertFalse(any("native.pose_meta" in error for error in result["errors"]))
        self.assertFalse(any("plugin.force_idr" in error for error in result["errors"]))
        self.assertFalse(any("plugin.orderly_eos" in error for error in result["errors"]))

    def test_v3dt_profile_declares_segmentation_and_both_tracker_internal_engines(self) -> None:
        result = validator.validate_manifest(self.manifest, check_files=True, profile="v3dt")
        self.assertFalse(result["ok"])
        blockers = "\n".join(result["blockers"])
        self.assertIn("engine.yolo26_seg_s", blockers)
        self.assertIn("engine.v3dt_bodypose", blockers)
        self.assertIn("engine.v3dt_tracker_reid", blockers)
        self.assertNotIn("engine.yolo26_detect_m", blockers)
        self.assertNotIn("engine.depth_tracking_dav2", blockers)

    def test_yolo11_seg_is_explicit_alternate_not_canonical(self) -> None:
        canonical = validator.validate_manifest(
            self.manifest, check_files=True, profile="canonical"
        )
        alternate = validator.validate_manifest(
            self.manifest, check_files=True, profile="yolo11_seg"
        )
        self.assertFalse(
            any("engine.yolo11_seg_alternate" in row for row in canonical["blockers"])
        )
        self.assertTrue(
            any("engine.yolo11_seg_alternate" in row for row in alternate["blockers"])
        )

    def test_provenance_gate_detects_declared_source_tampering(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        pose_meta = next(artifact for artifact in manifest["artifacts"] if artifact["id"] == "native.pose_meta")
        pose_meta["provenance"]["source_sha256"] = "0" * 64
        result = validator.validate_manifest(
            manifest,
            check_files=True,
            profile="canonical",
            require_provenance=True,
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("native.pose_meta: declared source SHA-256 mismatch" in error for error in result["errors"]),
            result["errors"],
        )

    def test_provenance_gate_rejects_ambiguous_wildcard_outputs(self) -> None:
        pose_meta = next(
            artifact
            for artifact in self.manifest["artifacts"]
            if artifact["id"] == "native.pose_meta"
        )
        original_matches = validator._matches

        def ambiguous_matches(raw: str, *, artifact_root: Path | None = None):
            matches = original_matches(raw, artifact_root=artifact_root)
            if raw == pose_meta["output"] and matches:
                return [*matches, matches[0].with_name("duplicate-pose-meta.so")]
            return matches

        with mock.patch.object(validator, "_matches", side_effect=ambiguous_matches):
            result = validator.validate_manifest(
                self.manifest,
                check_files=True,
                profile="canonical",
                require_provenance=True,
            )

        self.assertFalse(result["ok"])
        self.assertTrue(
            any(
                "native.pose_meta: multiple outputs match reviewed path" in error
                for error in result["errors"]
            ),
            result["errors"],
        )

    def test_structural_validation_does_not_require_machine_local_outputs(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        selected = next(
            row for row in manifest["artifacts"] if row["id"] == "engine.yolo26_detect_m"
        )
        selected["state"] = "staged_unverified"
        selected["required_profiles"] = ["fresh_clone_fixture"]
        with tempfile.TemporaryDirectory() as raw_root:
            artifact_root = Path(raw_root).resolve()
            structural = validator.validate_manifest(
                manifest,
                check_files=False,
                profile="fresh_clone_fixture",
                artifact_root=artifact_root,
            )
            files = validator.validate_manifest(
                manifest,
                check_files=True,
                profile="fresh_clone_fixture",
                artifact_root=artifact_root,
            )
        self.assertTrue(structural["ok"], structural)
        self.assertFalse(files["ok"], files)
        self.assertTrue(
            any("engine.yolo26_detect_m" in row for row in files["errors"] + files["blockers"]),
            files,
        )

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX package is required")
    def test_source_hash_covers_onnx_external_tensor_data(self) -> None:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            relative = Path("DS9/models/onnx/external.onnx")
            model_path = root / relative
            model_path.parent.mkdir(parents=True)
            weights = numpy_helper.from_array(np.arange(16, dtype=np.float32), name="weights")
            graph = helper.make_graph(
                [helper.make_node("Identity", ["weights"], ["output"])],
                "external-source-hash",
                [],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [16])],
                [weights],
            )
            model = helper.make_model(graph)
            onnx.save_model(
                model,
                str(model_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            sidecar = model_path.parent / "weights.bin"
            self.assertTrue(sidecar.is_file())
            with mock.patch.object(validator, "REPO_ROOT", root):
                before = validator._hash_sources([relative.as_posix()])
                with sidecar.open("ab") as handle:
                    handle.write(b"tamper")
                after = validator._hash_sources([relative.as_posix()])
            self.assertNotEqual(before, after)

    def test_generated_output_cannot_escape_ds9(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"][0]["output"] = "models/engines/forbidden.engine"
        manifest["artifacts"][0]["state"] = "missing"
        result = validator.validate_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("not below a declared DS9 output root" in error for error in result["errors"]))

    def test_builder_cannot_delegate_to_root_ds8_scripts(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["artifacts"][0]["builder"] = "scripts/build_native_extensions.sh"
        result = validator.validate_manifest(manifest)
        self.assertFalse(result["ok"])
        self.assertTrue(any("builder is not DS9-owned" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
