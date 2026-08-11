from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


DS9_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = DS9_ROOT / "scripts" / "build_rfdetr_1_8_3_engines.py"
SPEC = importlib.util.spec_from_file_location("rfdetr_1_8_3_engine_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class RFDETR183EngineBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = builder._load_matrix()
        self.release = dict(self.matrix["release"])

    def test_selection_is_explicit_and_rejects_licensed_rows(self) -> None:
        with self.assertRaisesRegex(builder.BuilderError, "explicit --model"):
            builder._select_models(self.matrix, [])

        selected = builder._select_models(
            self.matrix,
            ["detect_nano,seg_nano", "keypoint_preview"],
        )
        self.assertEqual(
            [row["id"] for row in selected],
            ["detect_nano", "seg_nano", "keypoint_preview"],
        )
        with self.assertRaisesRegex(builder.BuilderError, "licensed/disabled"):
            builder._select_models(self.matrix, ["detect_xlarge"])

    def test_all_open_rows_are_versioned_and_unambiguous(self) -> None:
        open_ids = [
            str(row["id"]) for row in self.matrix["models"] if row["enabled"]
        ]
        self.assertEqual(len(open_ids), 11)
        selected = builder._select_models(self.matrix, open_ids)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                builder._artifact_paths(root, model) for model in selected
            ]
        self.assertEqual(len({path.engine.name for path in paths}), len(paths))
        self.assertTrue(
            all(
                "/models/engines/rfdetr/1.8.3/" in str(path.engine)
                for path in paths
            )
        )

    def test_static_b3_source_receipt_is_required(self) -> None:
        model = next(
            row for row in self.matrix["models"] if row["id"] == "detect_nano"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = builder._artifact_paths(root, model)
            paths.onnx.parent.mkdir(parents=True)
            paths.onnx_receipt.parent.mkdir(parents=True)
            paths.onnx.write_bytes(b"reviewed-static-b3-onnx")
            digest = hashlib.sha256(paths.onnx.read_bytes()).hexdigest()
            receipt = {
                "schema": builder.SOURCE_PROVENANCE_SCHEMA,
                "artifact_kind": "onnx",
                "model_id": model["id"],
                "family": model["family"],
                "variant": model["variant"],
                "license": model["license"],
                "filename": paths.onnx.name,
                "release": self.release,
                "size_bytes": paths.onnx.stat().st_size,
                "sha256": digest,
                "checkpoint": {
                    "filename": model["checkpoint_filename"],
                    "md5": model["checkpoint_md5"],
                    "sha256": "a" * 64,
                },
                "tensor_contract": {
                    **builder._expected_tensor_contract(model, self.release),
                    "opset_imports": {"ai.onnx": 17},
                },
                "export": {
                    "class_name": model["class_name"],
                    "device": "cpu",
                    "batch_size": 3,
                    "dynamic_batch": False,
                    "format": "onnx",
                    "opset": 17,
                    "resolution": [model["resolution"], model["resolution"]],
                    "preserve_official_query_count": True,
                },
                "export_environment": {
                    "rfdetr": "1.8.3",
                    "rfdetr_source_commit": self.release["git_commit"],
                    "transformers": "5.14.1",
                    "onnx": "1.19.1",
                },
            }
            paths.onnx_receipt.write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            validated = builder._validate_onnx_receipt(
                paths, model, self.release
            )
            self.assertEqual(validated["sha256"], digest)
            receipt["export"]["dynamic_batch"] = True
            paths.onnx_receipt.write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with self.assertRaisesRegex(builder.BuilderError, "static-B3"):
                builder._validate_onnx_receipt(paths, model, self.release)

    def test_container_boundary_is_read_only_and_gpu_is_explicit(self) -> None:
        gpu = builder._container_security_args(gpu=True)
        self.assertIn("--network=none", gpu)
        self.assertIn("--read-only", gpu)
        self.assertIn("--cap-drop=ALL", gpu)
        self.assertIn("--security-opt=no-new-privileges", gpu)
        self.assertIn("--runtime=nvidia", gpu)
        self.assertIn("device=0", gpu)

        plan = builder._container_security_args(gpu=False)
        self.assertIn("--runtime=runc", plan)
        self.assertIn("NVIDIA_VISIBLE_DEVICES=void", plan)
        self.assertNotIn("--gpus", plan)

    def test_build_contract_is_static_b3_fp16(self) -> None:
        command = builder._inner_build_command("candidate.engine")
        self.assertEqual(
            command,
            (
                "trtexec",
                "--onnx=/inputs/model.onnx",
                "--fp16",
                "--memPoolSize=workspace:4096",
                "--saveEngine=/work/candidate.engine",
                "--skipInference",
            ),
        )
        self.assertIn("--fp16", command)
        self.assertIn("--memPoolSize=workspace:4096", command)
        self.assertIn("--skipInference", command)
        self.assertNotIn("--minShapes", " ".join(command))
        self.assertNotIn("--optShapes", " ".join(command))
        self.assertNotIn("--maxShapes", " ".join(command))

    def test_precision_canary_profiles_are_explicit_and_isolated(self) -> None:
        model = next(
            row
            for row in self.matrix["models"]
            if row["id"] == "seg_medium"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = builder._artifact_paths(root, model)
            observed: set[Path] = set()
            for profile_id in sorted(builder.CANARY_PRECISION_PROFILES):
                profile = builder._precision_profile(profile_id)
                assert profile is not None
                paths = builder._precision_canary_artifact_paths(
                    root, model, profile
                )
                self.assertNotEqual(paths.engine, baseline.engine)
                self.assertNotEqual(
                    paths.engine_receipt, baseline.engine_receipt
                )
                self.assertIn(
                    f"/canaries/{profile_id}/", str(paths.engine)
                )
                self.assertIn(
                    f"/canaries/{profile_id}/",
                    str(paths.engine_receipt),
                )
                self.assertTrue(paths.engine.name.endswith(".engine"))
                self.assertIn(profile_id, paths.engine.name)
                observed.add(paths.engine)
            self.assertEqual(
                len(observed), len(builder.CANARY_PRECISION_PROFILES)
            )

    def test_runtime_engine_profiles_are_explicit_and_collision_proof(
        self,
    ) -> None:
        model = next(
            row
            for row in self.matrix["models"]
            if row["id"] == "seg_medium"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = builder._artifact_paths(root, model)
            observed: set[Path] = set()
            for profile_id in builder.RUNTIME_ENGINE_PROFILES:
                paths = builder._runtime_engine_artifact_paths(
                    root, model, profile_id
                )
                self.assertNotEqual(paths.onnx, baseline.onnx)
                self.assertNotEqual(paths.engine, baseline.engine)
                self.assertNotEqual(
                    paths.engine_receipt, baseline.engine_receipt
                )
                self.assertIn(
                    f"/models/engines/rfdetr/1.8.3/runtime/{profile_id}/",
                    str(paths.engine),
                )
                self.assertIn(
                    f"/models/provenance/rfdetr/1.8.3/runtime/{profile_id}/",
                    str(paths.engine_receipt),
                )
                self.assertEqual(
                    paths.onnx.name,
                    model["runtime"]["onnx_filename"],
                )
                self.assertEqual(
                    paths.onnx_receipt.name,
                    "seg_medium.runtime_onnx.sub_div_float32_v1.json",
                )
                self.assertEqual(
                    paths.engine.name,
                    (
                        Path(model["runtime"]["onnx_filename"]).stem
                        + f"_{profile_id}.engine"
                    ),
                )
                observed.add(paths.engine)
            self.assertEqual(
                len(observed), len(builder.RUNTIME_ENGINE_PROFILES)
            )

    def test_runtime_profile_cli_is_mutually_exclusive_and_needs_no_baseline(
        self,
    ) -> None:
        args = builder._parse_args(
            [
                "--model",
                "detect_nano",
                "--runtime-engine-profile",
                builder.BASELINE_PRECISION_PROFILE,
            ]
        )
        self.assertEqual(
            args.runtime_engine_profile,
            builder.BASELINE_PRECISION_PROFILE,
        )
        self.assertEqual(args.precision_canary_profile, "")
        with self.assertRaises(SystemExit):
            builder._parse_args(
                [
                    "--model",
                    "detect_nano",
                    "--precision-canary-profile",
                    "fp32_no_tf32",
                    "--runtime-engine-profile",
                    "fp32_no_tf32",
                ]
            )

        baseline = builder._runtime_build_contract(
            next(
                row
                for row in self.matrix["models"]
                if row["id"] == "detect_nano"
            ),
            builder.BASELINE_PRECISION_PROFILE,
            None,
        )
        self.assertEqual(baseline["profile"], "fp16_tf32")
        self.assertEqual(
            baseline["trtexec_precision_args"], ["--fp16"]
        )
        self.assertFalse(baseline["layer_info_exported"])
        nonbaseline_profile = builder._precision_profile("fp32_no_tf32")
        assert nonbaseline_profile is not None
        nonbaseline = builder._runtime_build_contract(
            next(
                row
                for row in self.matrix["models"]
                if row["id"] == "detect_nano"
            ),
            "fp32_no_tf32",
            nonbaseline_profile,
        )
        self.assertEqual(
            nonbaseline["trtexec_precision_args"], ["--noTF32"]
        )
        self.assertTrue(nonbaseline["layer_info_exported"])
        self.assertNotEqual(
            builder.RUNTIME_ENGINE_RECEIPT_SCHEMA,
            builder.ENGINE_RECEIPT_SCHEMA,
        )
        self.assertNotEqual(
            builder.RUNTIME_ENGINE_RECEIPT_SCHEMA,
            builder.CANARY_ENGINE_RECEIPT_SCHEMA,
        )

    def test_runtime_onnx_receipt_binds_exact_adapter_and_source(self) -> None:
        model = next(
            row
            for row in self.matrix["models"]
            if row["id"] == "detect_nano"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = builder._artifact_paths(root, model)
            runtime = builder._runtime_engine_artifact_paths(
                root,
                model,
                builder.BASELINE_PRECISION_PROFILE,
            )
            baseline.onnx.parent.mkdir(parents=True)
            baseline.onnx_receipt.parent.mkdir(parents=True)
            runtime.onnx.parent.mkdir(parents=True)
            baseline.onnx.write_bytes(b"reviewed-static-b3-onnx")
            baseline_digest = hashlib.sha256(
                baseline.onnx.read_bytes()
            ).hexdigest()
            tensor_contract = {
                **builder._expected_tensor_contract(
                    model, self.release
                ),
                "opset_imports": {"ai.onnx": 17},
            }
            source_receipt = {
                "schema": builder.SOURCE_PROVENANCE_SCHEMA,
                "artifact_kind": "onnx",
                "model_id": model["id"],
                "family": model["family"],
                "variant": model["variant"],
                "license": model["license"],
                "filename": baseline.onnx.name,
                "release": self.release,
                "size_bytes": baseline.onnx.stat().st_size,
                "sha256": baseline_digest,
                "checkpoint": {
                    "filename": model["checkpoint_filename"],
                    "md5": model["checkpoint_md5"],
                    "sha256": "a" * 64,
                },
                "tensor_contract": tensor_contract,
                "export": {
                    "class_name": model["class_name"],
                    "device": "cpu",
                    "batch_size": 3,
                    "dynamic_batch": False,
                    "format": "onnx",
                    "opset": 17,
                    "resolution": [
                        model["resolution"],
                        model["resolution"],
                    ],
                    "preserve_official_query_count": True,
                },
                "export_environment": {
                    "rfdetr": "1.8.3",
                    "rfdetr_source_commit": self.release["git_commit"],
                    "transformers": "5.14.1",
                    "onnx": "1.19.1",
                },
            }
            baseline.onnx_receipt.write_text(
                json.dumps(source_receipt), encoding="utf-8"
            )
            runtime.onnx.write_bytes(b"reviewed-runtime-adapter-onnx")
            runtime_digest = hashlib.sha256(
                runtime.onnx.read_bytes()
            ).hexdigest()
            runtime_contract = {
                **tensor_contract,
                "runtime_adapter": json.loads(
                    json.dumps(builder.RUNTIME_ADAPTER_SPEC)
                ),
                "normalized_input_consumer_count": 1,
            }
            runtime_receipt = {
                "schema": builder.SOURCE_PROVENANCE_SCHEMA,
                "artifact_kind": "runtime_onnx",
                "model_id": model["id"],
                "family": model["family"],
                "variant": model["variant"],
                "license": model["license"],
                "release": self.release,
                "filename": runtime.onnx.name,
                "size_bytes": runtime.onnx.stat().st_size,
                "sha256": runtime_digest,
                "source": {
                    "filename": baseline.onnx.name,
                    "size_bytes": baseline.onnx.stat().st_size,
                    "sha256": baseline_digest,
                    "receipt": baseline.onnx_receipt.name,
                    "receipt_sha256": hashlib.sha256(
                        baseline.onnx_receipt.read_bytes()
                    ).hexdigest(),
                },
                "adapter": json.loads(
                    json.dumps(builder.RUNTIME_ADAPTER_SPEC)
                ),
                "tensor_contract": runtime_contract,
            }
            runtime.onnx_receipt.write_text(
                json.dumps(runtime_receipt), encoding="utf-8"
            )
            validated = builder._validate_runtime_onnx_receipt(
                root, runtime, model, self.release
            )
            self.assertEqual(validated["sha256"], runtime_digest)
            self.assertEqual(
                validated["adapter"], builder.RUNTIME_ADAPTER_SPEC
            )
            self.assertEqual(
                validated["normalized_source"]["sha256"],
                baseline_digest,
            )

            runtime_receipt["adapter"]["std"][0] = 0.230
            runtime.onnx_receipt.write_text(
                json.dumps(runtime_receipt), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                builder.BuilderError, "adapter contract"
            ):
                builder._validate_runtime_onnx_receipt(
                    root, runtime, model, self.release
                )

    def test_precision_canary_commands_match_exact_trt_10_14_modes(
        self,
    ) -> None:
        model = next(
            row
            for row in self.matrix["models"]
            if row["id"] == "seg_medium"
        )
        expected = {
            "fp32_tf32": (),
            "fp32_no_tf32": ("--noTF32",),
            "fp16_no_tf32": ("--fp16", "--noTF32"),
        }
        for profile_id, flags in expected.items():
            with self.subTest(profile=profile_id):
                profile = builder._precision_profile(profile_id)
                assert profile is not None
                command = builder._inner_build_command(
                    "candidate.engine",
                    model=model,
                    precision_profile=profile,
                    layer_info_name="layers.json",
                )
                for flag in flags:
                    self.assertIn(flag, command)
                if "--fp16" not in flags:
                    self.assertNotIn("--fp16", command)
                self.assertIn("--profilingVerbosity=detailed", command)
                self.assertIn("--dumpLayerInfo", command)
                self.assertIn(
                    "--exportLayerInfo=/work/layers.json", command
                )
                self.assertNotIn("--stronglyTyped", command)

    def test_selective_canary_obeys_verified_fp32_head_patterns(
        self,
    ) -> None:
        profile = builder._precision_profile(
            "fp16_fp32_heads_no_tf32"
        )
        assert profile is not None
        for model_id, required_fragment in (
            ("detect_medium", "/class_embed/*:fp32"),
            (
                "seg_medium",
                "/segmentation_head/Einsum:fp32",
            ),
            (
                "keypoint_preview",
                "/transformer/decoder/layers.3/kp_norm5/*:fp32",
            ),
        ):
            with self.subTest(model=model_id):
                model = next(
                    row
                    for row in self.matrix["models"]
                    if row["id"] == model_id
                )
                command = builder._inner_build_command(
                    "candidate.engine",
                    model=model,
                    precision_profile=profile,
                    layer_info_name="layers.json",
                )
                joined = " ".join(command)
                self.assertIn("--precisionConstraints=obey", command)
                self.assertIn("--fp16", command)
                self.assertIn("--noTF32", command)
                self.assertIn(required_fragment, joined)
                contract = builder._build_contract(model, profile)
                self.assertEqual(
                    contract["profile"],
                    "fp16_fp32_heads_no_tf32",
                )
                self.assertEqual(
                    contract["precision_constraints"], "obey"
                )
                self.assertTrue(contract["layer_info_exported"])
                self.assertTrue(contract["layer_precisions"])

    def test_precision_canary_requires_layer_info_and_valid_profile(
        self,
    ) -> None:
        model = next(
            row
            for row in self.matrix["models"]
            if row["id"] == "detect_medium"
        )
        profile = builder._precision_profile("fp32_no_tf32")
        assert profile is not None
        with self.assertRaisesRegex(
            builder.BuilderError, "layer-info filename"
        ):
            builder._inner_build_command(
                "candidate.engine",
                model=model,
                precision_profile=profile,
            )
        with self.assertRaisesRegex(
            builder.BuilderError, "unknown precision canary"
        ):
            builder._precision_profile("fast-and-loose")

    def test_layer_info_evidence_is_validated_and_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.layers.json"
            target_dir = root / "evidence"
            target_dir.mkdir(mode=0o700)
            target = target_dir / "layer-info.json"
            source.write_text(
                json.dumps({"Layers": [{"Name": "head"}]}),
                encoding="utf-8",
            )
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            record = builder._install_layer_info(source, target)
            self.assertFalse(source.exists())
            self.assertTrue(target.is_file())
            self.assertEqual(record["sha256"], expected)
            self.assertEqual(record["path"], target)
            self.assertEqual(record["mode"], "0600")

            invalid = root / "invalid.layers.json"
            invalid.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(
                builder.BuilderError, "valid JSON"
            ):
                builder._install_layer_info(
                    invalid, target_dir / "invalid.json"
                )

    def test_plan_rejects_writable_output_destinations(self) -> None:
        model = next(
            row for row in self.matrix["models"] if row["id"] == "detect_nano"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = builder._artifact_paths(root, model)
            paths.engine.parent.mkdir(parents=True)
            paths.engine_receipt.parent.mkdir(parents=True)
            paths.engine.parent.chmod(0o775)
            with self.assertRaisesRegex(
                builder.BuilderError, "engine output directory"
            ):
                builder._validate_output_destinations(root, [model])
            paths.engine.parent.chmod(0o755)
            paths.engine_receipt.parent.chmod(0o775)
            with self.assertRaisesRegex(
                builder.BuilderError, "engine provenance directory"
            ):
                builder._validate_output_destinations(root, [model])
            paths.engine_receipt.parent.chmod(0o755)
            builder._validate_output_destinations(root, [model])

    def test_deserialize_transcript_requires_positive_proof(self) -> None:
        good = """
        Loaded engine size: 42 MiB
        Engine deserialized in 0.123 sec.
        Skipped inference phase since --skipInference is added.
        &&&& PASSED TensorRT.trtexec
        """
        builder._validate_trtexec_transcript(good, require_load=True)
        with self.assertRaisesRegex(builder.BuilderError, "deserialized"):
            builder._validate_trtexec_transcript(
                good.replace("Engine deserialized in 0.123 sec.", ""),
                require_load=True,
            )


if __name__ == "__main__":
    unittest.main()
