from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


DS9_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = DS9_ROOT / "scripts" / "validate_rfdetr_1_8_3_media.py"
SPEC = importlib.util.spec_from_file_location(
    "rfdetr_1_8_3_media_validation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _runner_manifest(
    model: dict[str, object], case_name: str = "batch-180"
) -> dict[str, object]:
    resolution = int(model["resolution"])
    outputs = dict(model["outputs"])
    input_shape = [3, 3, resolution, resolution]
    tensor_rows: list[dict[str, object]] = [
        {
            "name": "input",
            "mode": "input",
            "location": "device",
            "data_type": "fp32",
            "shape": input_shape,
            "element_count": math.prod(input_shape),
            "byte_count": math.prod(input_shape) * 4,
        }
    ]
    output_rows: list[dict[str, object]] = []
    for name, raw_shape in outputs.items():
        shape = [int(value) for value in raw_shape]
        element_count = math.prod(shape)
        tensor_rows.append(
            {
                "name": name,
                "mode": "output",
                "location": "device",
                "data_type": "fp32",
                "shape": shape,
                "element_count": element_count,
                "byte_count": element_count * 4,
            }
        )
        output_rows.append(
            {
                "name": name,
                "shape": shape,
                "byte_count": element_count * 4,
                "file": f"{case_name}__{name}.bin",
                "repeat_stable": True,
                "finite": True,
                "min": -1.0,
                "max": 1.0,
            }
        )
    return {
        "schema": "noesis.rfdetr_1_8_3.media_runner.v1",
        "case": case_name,
        "repeat_stable": True,
        "inference_elapsed_ms": {"first": 1.25, "repeat": 1.0},
        "tensors": tensor_rows,
        "input": {
            "name": "input",
            "shape": input_shape,
            "byte_count": math.prod(input_shape) * 4,
            "finite": True,
            "min": -2.2,
            "max": 2.6,
        },
        "outputs": output_rows,
    }


class RFDETR183MediaValidationTests(unittest.TestCase):
    def test_sigmoid_is_stable_at_large_magnitudes(self) -> None:
        positive = float(validator._sigmoid(1000.0))
        negative = float(validator._sigmoid(-1000.0))
        self.assertTrue(math.isfinite(positive))
        self.assertTrue(math.isfinite(negative))
        self.assertGreater(positive, 1.0 - 1e-15)
        self.assertGreater(negative, 0.0)
        self.assertLess(negative, 1e-37)
        self.assertAlmostEqual(float(validator._sigmoid(0.0)), 0.5)
        self.assertAlmostEqual(
            float(validator._sigmoid(math.log(3.0))), 0.75, places=12
        )

    def test_keypoint_score_components_keep_base_score_distinct(self) -> None:
        import numpy as np

        labels = np.zeros((1, 1, 2), dtype=np.float32)
        keypoints = np.zeros((1, 1, 34, 8), dtype=np.float32)
        base, fused = validator._keypoint_score_components(
            labels, keypoints
        )
        self.assertAlmostEqual(float(base[0, 0]), 0.5)
        self.assertAlmostEqual(
            float(fused[0, 0]),
            0.5 * (2.0 ** -0.2),
            places=12,
        )
        self.assertNotEqual(float(base[0, 0]), float(fused[0, 0]))

    def test_cxcywh_iou_covers_identity_overlap_and_disjoint_boxes(self) -> None:
        self.assertAlmostEqual(
            validator._cxcywh_iou(
                [0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]
            ),
            1.0,
        )
        self.assertAlmostEqual(
            validator._cxcywh_iou(
                [0.5, 0.5, 0.4, 0.4], [0.6, 0.5, 0.4, 0.4]
            ),
            0.6,
        )
        self.assertEqual(
            validator._cxcywh_iou(
                [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]
            ),
            0.0,
        )
        self.assertEqual(
            validator._cxcywh_iou(
                [0.5, 0.5, 0.0, 0.4], [0.5, 0.5, 0.4, 0.4]
            ),
            0.0,
        )
        with self.assertRaisesRegex(
            validator.ValidationError, "four coordinates"
        ):
            validator._cxcywh_iou([0.5, 0.5, 0.4], [0.5, 0.5, 0.4, 0.4])

    def test_semantic_matcher_prefers_retained_true_candidate_over_iou(
        self,
    ) -> None:
        reference = [[0.50, 0.50, 0.40, 0.40]]
        candidate = [
            [0.50, 0.50, 0.40, 0.40],
            [0.505, 0.50, 0.40, 0.40],
            [0.515, 0.50, 0.40, 0.40],
        ]
        assignments = validator._match_queries_semantically(
            reference,
            candidate,
            [1],
            [2, 1, 1],
            [[0.99, 0.69, 0.90]],
        )
        self.assertEqual(assignments[0]["candidate_index"], 2)
        self.assertTrue(assignments[0]["semantic_retained_edge"])
        self.assertLess(
            assignments[0]["iou"],
            validator._cxcywh_iou(reference[0], candidate[0]),
        )

    def test_semantic_matcher_preserves_real_below_threshold_retention(
        self,
    ) -> None:
        reference = [[0.50, 0.50, 0.40, 0.40]]
        candidate = [
            [0.50, 0.50, 0.40, 0.40],
            [0.51, 0.50, 0.40, 0.40],
            [0.05, 0.05, 0.04, 0.04],
        ]
        assignments = validator._match_queries_semantically(
            reference,
            candidate,
            [1],
            [2, 1, 1],
            [[0.10, 0.69, 0.99]],
        )
        match = assignments[0]
        self.assertEqual(match["candidate_index"], 1)
        self.assertTrue(match["spatial_match"])
        self.assertTrue(match["class_match"])
        self.assertFalse(match["score_retained"])
        self.assertFalse(match["semantic_retained_edge"])

        nonspatial = validator._match_queries_semantically(
            reference,
            [candidate[0], candidate[2]],
            [1],
            [2, 1],
            [[0.10, 0.99]],
        )[0]
        self.assertEqual(nonspatial["candidate_index"], 0)
        self.assertTrue(nonspatial["spatial_match"])
        self.assertFalse(nonspatial["semantic_class_edge"])
        self.assertFalse(nonspatial["semantic_retained_edge"])

    def test_semantic_matcher_is_permutation_safe_unique_and_dominant(
        self,
    ) -> None:
        reference = [
            [0.20, 0.20, 0.10, 0.10],
            [0.80, 0.20, 0.10, 0.10],
            [0.50, 0.80, 0.10, 0.10],
        ]
        candidate = [
            reference[2],
            reference[0],
            [0.50, 0.50, 0.05, 0.05],
            reference[1],
        ]
        assignments = validator._match_queries_semantically(
            reference,
            candidate,
            [1, 2, 3],
            [3, 1, 9, 2],
            [
                [0.10, 0.90, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.90],
                [0.90, 0.10, 0.10, 0.10],
            ],
        )
        self.assertEqual(
            [row["candidate_index"] for row in assignments],
            [1, 3, 0],
        )
        self.assertEqual(
            len({row["candidate_index"] for row in assignments}), 3
        )
        weights = validator._semantic_assignment_weights(3)
        self.assertGreater(weights["spatial_match"], 3 * weights["iou"])
        self.assertGreater(
            weights["correct_class"],
            3 * (weights["spatial_match"] + weights["iou"]),
        )
        self.assertGreater(
            weights["retained_correct"],
            3
            * (
                weights["correct_class"]
                + weights["spatial_match"]
                + weights["iou"]
            ),
        )

    def test_semantic_matcher_rejects_invalid_contracts(self) -> None:
        reference = [[0.50, 0.50, 0.40, 0.40]]
        candidate = [[0.50, 0.50, 0.40, 0.40]]
        invalid_calls = [
            (reference * 2, candidate, [1, 1], [1], [[0.9], [0.9]], {}),
            (reference, candidate, [1], [1], [[float("nan")]], {}),
            (reference, candidate, [1.0], [1], [[0.9]], {}),
            (reference, [[0.5, 0.5, -0.1, 0.4]], [1], [1], [[0.9]], {}),
            (reference, candidate, [1], [1], [[0.9]], {"iou_floor": 1.1}),
        ]
        for args in invalid_calls:
            with self.subTest(args=args):
                with self.assertRaises(validator.ValidationError):
                    validator._match_queries_semantically(
                        *args[:5], **args[5]
                    )

    def test_person_matcher_uses_person_channel_and_is_symmetric(self) -> None:
        reference_boxes = [
            [0.20, 0.20, 0.10, 0.10],
            [0.80, 0.20, 0.10, 0.10],
        ]
        candidate_boxes = [
            [0.80, 0.20, 0.10, 0.10],
            [0.20, 0.20, 0.10, 0.10],
            [0.50, 0.80, 0.10, 0.10],
        ]
        result = validator._match_person_prediction_sets(
            reference_boxes,
            candidate_boxes,
            [0.90, 0.80],
            [0.80, 0.90, 0.70],
        )
        spatial = [
            row for row in result["matches"] if row["spatial_match"]
        ]
        self.assertEqual(
            [
                (row["reference_query"], row["candidate_query"])
                for row in spatial
            ],
            [(0, 1), (1, 0)],
        )
        self.assertEqual(result["unmatched_reference_queries"], [])
        self.assertEqual(result["unmatched_candidate_queries"], [2])

        dropped = validator._match_person_prediction_sets(
            reference_boxes,
            candidate_boxes,
            [0.90, 0.80],
            [0.80, 0.49, 0.10],
        )
        self.assertEqual(dropped["unmatched_reference_queries"], [0])
        self.assertEqual(dropped["unmatched_candidate_queries"], [])

    def test_person_matcher_handles_empty_thresholded_sides(self) -> None:
        result = validator._match_person_prediction_sets(
            [[0.20, 0.20, 0.10, 0.10]],
            [[0.20, 0.20, 0.10, 0.10]],
            [0.49],
            [0.80],
        )
        self.assertEqual(result["reference_queries"], [])
        self.assertEqual(result["candidate_queries"], [0])
        self.assertEqual(result["matches"], [])
        self.assertEqual(result["unmatched_reference_queries"], [])
        self.assertEqual(result["unmatched_candidate_queries"], [0])

    def test_successor_report_lineage_is_explicit_and_digest_bound(
        self,
    ) -> None:
        predecessor = {
            "schema": "noesis.ds9.rfdetr-media-quality-report.v1",
            "run_id": "run-1",
            "recorded_at_utc": "2026-07-24T00:00:00Z",
            "promotion_status": "unpromoted",
            "runtime_selected": False,
            "automated_status": "failed",
        }
        predecessor["manifest_sha256"] = validator._json_digest(predecessor)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            path = run_dir / "report-matched-v2.json"
            path.write_text(
                json.dumps(predecessor, sort_keys=True),
                encoding="utf-8",
            )
            lineage = validator._report_successor_lineage(
                run_dir,
                "run-1",
                "report-semantic-v3.json",
                path.name,
            )
            assert lineage is not None
            self.assertEqual(lineage["path"], path.name)
            self.assertEqual(lineage["sha256"], validator._sha256(path))
            self.assertEqual(
                lineage["manifest_sha256"],
                predecessor["manifest_sha256"],
            )
            self.assertEqual(
                lineage["reason"],
                "quality_scope_and_keypoint_score_correction",
            )
            with self.assertRaisesRegex(
                validator.ValidationError, "cannot supersede itself"
            ):
                validator._report_successor_lineage(
                    run_dir,
                    "run-1",
                    path.name,
                    path.name,
                )
            with self.assertRaisesRegex(
                validator.ValidationError, "contract drifted"
            ):
                validator._report_successor_lineage(
                    run_dir,
                    "different-run",
                    "report-semantic-v3.json",
                    path.name,
                )

    def test_percentiles_use_deterministic_linear_interpolation(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0, 40.0]
        self.assertEqual(validator._percentile(values, 0.0), 0.0)
        self.assertEqual(validator._percentile(values, 50.0), 20.0)
        self.assertEqual(validator._percentile(values, 100.0), 40.0)
        self.assertAlmostEqual(validator._percentile(values, 5.0), 2.0)
        self.assertAlmostEqual(validator._percentile(values, 95.0), 38.0)
        summary = validator._summarize_distribution(values)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["min"], 0.0)
        self.assertEqual(summary["p05"], 2.0)
        self.assertEqual(summary["median"], 20.0)
        self.assertEqual(summary["p95"], 38.0)
        self.assertEqual(summary["max"], 40.0)
        self.assertEqual(summary["mean"], 20.0)
        with self.assertRaisesRegex(
            validator.ValidationError, "finite values"
        ):
            validator._percentile([], 50.0)
        with self.assertRaisesRegex(
            validator.ValidationError, r"\[0,100\]"
        ):
            validator._percentile(values, 100.01)

    def test_quality_gate_constants_and_evaluation_match_reviewed_contract(
        self,
    ) -> None:
        self.assertEqual(
            validator.DETECTION_GATES,
            {
                "class_agreement_min": 1.0,
                "retention_min": 1.0,
                "box_iou_median_min": 0.98,
                "box_iou_p05_min": 0.90,
                "score_abs_error_p95_max": 0.03,
            },
        )
        self.assertEqual(
            validator.SEGMENTATION_GATES,
            {
                "mask_iou_median_min": 0.95,
                "mask_iou_p05_min": 0.85,
                "mask_area_relative_error_p95_max": 0.10,
            },
        )
        self.assertEqual(
            validator.PERSON_TASK_GATES,
            {
                "reference_retention_min": 1.0,
                "candidate_precision_min": 1.0,
                "box_iou_median_min": 0.98,
                "box_iou_p05_min": 0.90,
                "score_abs_error_p95_max": 0.03,
            },
        )
        self.assertEqual(
            validator.KEYPOINT_GATES,
            {
                "base_score_abs_error_p95_max": 0.03,
                "fused_score_relative_error_p95_max": 0.03,
                "oks_median_min": 0.97,
                "oks_min_min": 0.90,
                "coordinate_error_box_diagonal_p95_max": 0.01,
                "visible_jaccard_min": 0.90,
            },
        )
        passing = {
            "class_agreement": 1.0,
            "retention": 1.0,
            "box_iou": {"median": 0.99, "p05": 0.95},
            "score_abs_error": {"p95": 0.02},
            "mask_iou": {"median": 0.97, "p05": 0.90},
            "mask_area_relative_error": {"p95": 0.08},
        }
        results = validator._apply_quality_gates("segmentation", passing)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(row["passed"] for row in results))
        failing = json.loads(json.dumps(passing))
        failing["mask_iou"]["p05"] = 0.84
        results = validator._apply_quality_gates("segmentation", failing)
        failed_names = [row["name"] for row in results if not row["passed"]]
        self.assertEqual(failed_names, ["mask_iou_p05_min"])
        with self.assertRaisesRegex(
            validator.ValidationError, "missing quality metric"
        ):
            validator._apply_quality_gates("detection", {})

        person_passing = {
            "reference_retention": 1.0,
            "candidate_precision": 1.0,
            "box_iou": {"median": 0.99, "p05": 0.95},
            "score_abs_error": {"p95": 0.02},
        }
        person_results = validator._apply_quality_gates(
            "detection", person_passing, scope="person_task"
        )
        self.assertEqual(len(person_results), 5)
        self.assertTrue(all(row["passed"] for row in person_results))
        person_passing["candidate_precision"] = 0.99
        person_results = validator._apply_quality_gates(
            "detection", person_passing, scope="person_task"
        )
        self.assertEqual(
            [
                row["name"]
                for row in person_results
                if not row["passed"]
            ],
            ["candidate_precision_min"],
        )

    def test_keypoint_score_gates_split_base_and_fused_error(self) -> None:
        metrics = {
            "reference_retention": 1.0,
            "candidate_precision": 1.0,
            "box_iou": {"median": 0.99, "p05": 0.95},
            "base_score_abs_error": {"p95": 0.02},
            "fused_score_relative_error": {"p95": 0.02},
            "oks": {"median": 0.99, "min": 0.95},
            "coordinate_error_box_diagonal": {"p95": 0.005},
            "visible_jaccard": {"min": 1.0},
        }
        results = validator._apply_quality_gates(
            "keypoint", metrics, scope="person_task"
        )
        self.assertEqual(len(results), 10)
        self.assertTrue(all(row["passed"] for row in results))
        metrics["fused_score_relative_error"]["p95"] = 0.031
        results = validator._apply_quality_gates(
            "keypoint", metrics, scope="person_task"
        )
        self.assertEqual(
            [row["name"] for row in results if not row["passed"]],
            ["fused_score_relative_error_p95_max"],
        )
        with self.assertRaisesRegex(
            validator.ValidationError, "all-class"
        ):
            validator._apply_quality_gates(
                "keypoint", metrics, scope="all_classes"
            )

    def test_matrix_loader_rejects_output_contract_drift(self) -> None:
        original = json.loads(
            validator.MATRIX_PATH.read_text(encoding="utf-8")
        )
        model = next(
            row for row in original["models"] if row["id"] == "detect_nano"
        )
        model["outputs"]["dets"] = [3, 0, 4]
        with tempfile.TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "matrix.json"
            matrix_path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(
                validator.ValidationError, "tensor contract"
            ):
                validator._load_matrix(matrix_path)

    def test_successor_analysis_reuses_prepared_tensor_contract(self) -> None:
        prepared = {
            "models": [
                {
                    "id": "detect_nano",
                    "family": "detection",
                    "variant": "nano",
                    "resolution": 384,
                    "expected_input": [3, 3, 384, 384],
                    "expected_outputs": {
                        "dets": [3, 300, 4],
                        "labels": [3, 300, 91],
                    },
                }
            ]
        }
        models = validator._comparison_models_from_run(prepared)
        self.assertEqual(
            models,
            [
                {
                    "id": "detect_nano",
                    "family": "detection",
                    "variant": "nano",
                    "resolution": 384,
                    "outputs": {
                        "dets": [3, 300, 4],
                        "labels": [3, 300, 91],
                    },
                }
            ],
        )
        prepared["models"][0]["expected_input"] = [1, 3, 384, 384]
        with self.assertRaisesRegex(
            validator.ValidationError, "tensor contract"
        ):
            validator._comparison_models_from_run(prepared)

    def test_model_selection_is_explicit_and_rejects_disabled_rows(self) -> None:
        matrix = validator._load_matrix()
        with self.assertRaisesRegex(
            validator.ValidationError, "explicit --model"
        ):
            validator._select_models(matrix, [])
        selected = validator._select_models(
            matrix, ["detect_nano,seg_nano", "keypoint_preview"]
        )
        self.assertEqual(
            [row["id"] for row in selected],
            ["detect_nano", "seg_nano", "keypoint_preview"],
        )
        with self.assertRaisesRegex(
            validator.ValidationError, "disabled"
        ):
            validator._select_models(matrix, ["detect_xlarge"])

    def test_runtime_cli_modes_are_explicit_and_mutually_exclusive(
        self,
    ) -> None:
        self.assertIn(
            validator.BASELINE_RUNTIME_ENGINE_PROFILE,
            validator.RUNTIME_ENGINE_PROFILES,
        )
        for profile in validator.RUNTIME_ENGINE_PROFILES:
            with self.subTest(profile=profile):
                prepared = validator._parse_args(
                    [
                        "prepare",
                        "--model",
                        "detect_nano",
                        "--media-dir",
                        "/private/media",
                        "--runtime-engine-profile",
                        profile,
                    ]
                )
                self.assertEqual(prepared.runtime_engine_profile, profile)
                trt = validator._parse_args(
                    [
                        "trt",
                        "--run-dir",
                        "/private/run",
                        "--runtime-engine-profile",
                        profile,
                    ]
                )
                self.assertEqual(trt.runtime_engine_profile, profile)
                compared = validator._parse_args(
                    [
                        "compare",
                        "--run-dir",
                        "/private/run",
                        "--runtime-engine-profile",
                        profile,
                    ]
                )
                self.assertEqual(compared.runtime_engine_profile, profile)
        with self.assertRaises(SystemExit):
            validator._parse_args(
                [
                    "trt",
                    "--run-dir",
                    "/private/run",
                    "--precision-canary-profile",
                    "fp32_no_tf32",
                    "--runtime-engine-profile",
                    validator.BASELINE_RUNTIME_ENGINE_PROFILE,
                ]
            )

    def test_prepared_runtime_contract_requires_raw_rgb01_and_adapter(
        self,
    ) -> None:
        profile = validator.RUNTIME_ENGINE_PROFILES[0]
        run = {
            "artifact_role": validator.RUNTIME_VALIDATION_ARTIFACT_ROLE,
            "runtime_engine_profile": profile,
            "workflow": {
                "preprocess": {
                    "resize": "direct_square_bilinear",
                    "color": "RGB",
                    "scale": "uint8_to_float32_[0,1]",
                    "mean": None,
                    "std": None,
                    "letterbox": False,
                    "input_contract": validator.RUNTIME_INPUT_CONTRACT,
                    "runtime_adapter_revision": (
                        validator.RUNTIME_ADAPTER_REVISION
                    ),
                    "normalization_location": (
                        "revision_bound_runtime_onnx"
                    ),
                }
            },
            "models": [
                {
                    "onnx": {
                        "artifact_kind": "runtime_onnx",
                        "input_contract": validator.RUNTIME_INPUT_CONTRACT,
                        "adapter_revision": (
                            validator.RUNTIME_ADAPTER_REVISION
                        ),
                        "adapter": json.loads(
                            json.dumps(validator.RUNTIME_ADAPTER_SPEC)
                        ),
                    },
                    "cases": [
                        {
                            "input": {
                                "dtype": "float32",
                                "finite": True,
                                "min": 0.0,
                                "max": 1.0,
                            }
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            validator._prepared_runtime_profile(run), profile
        )
        drifted = json.loads(json.dumps(run))
        drifted["models"][0]["cases"][0]["input"]["max"] = 1.01
        with self.assertRaisesRegex(
            validator.ValidationError, "RGB01 tensor"
        ):
            validator._prepared_runtime_profile(drifted)
        drifted = json.loads(json.dumps(run))
        drifted["models"][0]["onnx"]["adapter"]["std"][0] = 0.230
        with self.assertRaisesRegex(
            validator.ValidationError, "ONNX/input contract"
        ):
            validator._prepared_runtime_profile(drifted)
        self.assertEqual(
            validator._prepared_runtime_profile(
                {"workflow": {"preprocess": {}}, "models": []}
            ),
            "",
        )

    def test_runtime_engine_receipt_is_profile_and_adapter_bound(self) -> None:
        real_builder = validator._builder_module()
        matrix = validator._load_matrix()
        model = next(
            row
            for row in matrix["models"]
            if row["id"] == "detect_nano"
        )
        profile_id = validator.RUNTIME_ENGINE_PROFILES[0]
        precision = real_builder._runtime_precision_profile(profile_id)
        assert precision is not None
        expected_input = [3, 3, model["resolution"], model["resolution"]]
        expected_outputs = model["outputs"]
        tensor_contract = {
            "input": {"input": expected_input},
            "outputs": expected_outputs,
            "runtime_adapter": real_builder.RUNTIME_ADAPTER_SPEC,
            "normalized_input_consumer_count": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = SimpleNamespace(
                onnx=root / "models/onnx/runtime/model.onnx",
                onnx_receipt=root
                / "models/provenance/model.runtime_onnx.json",
                engine=root / "models/engines/runtime/model.engine",
                engine_receipt=root
                / "models/provenance/runtime/model.engine.json",
            )
            normalized_path = root / "models/onnx/model.onnx"
            normalized_receipt = root / "models/provenance/model.onnx.json"
            for path, payload in (
                (paths.onnx, b"runtime-onnx"),
                (paths.onnx_receipt, b"runtime-receipt"),
                (paths.engine, b"runtime-engine"),
                (normalized_path, b"normalized-onnx"),
                (normalized_receipt, b"normalized-receipt"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            runtime_source = {
                "size_bytes": paths.onnx.stat().st_size,
                "sha256": validator._sha256(paths.onnx),
                "receipt_sha256": validator._sha256(
                    paths.onnx_receipt
                ),
                "tensor_contract": tensor_contract,
                "adapter": real_builder.RUNTIME_ADAPTER_SPEC,
                "normalized_source": {
                    "path": normalized_path,
                    "size_bytes": normalized_path.stat().st_size,
                    "sha256": validator._sha256(normalized_path),
                    "receipt_path": normalized_receipt,
                    "receipt_sha256": validator._sha256(
                        normalized_receipt
                    ),
                },
            }

            class FakeBuilder:
                RUNTIME_ENGINE_RECEIPT_SCHEMA = (
                    real_builder.RUNTIME_ENGINE_RECEIPT_SCHEMA
                )
                REQUIRED_TRT_VERSION = real_builder.REQUIRED_TRT_VERSION
                REQUIRED_IMAGE_ID = real_builder.REQUIRED_IMAGE_ID
                REQUIRED_IMAGE_REF = real_builder.REQUIRED_IMAGE_REF
                REQUIRED_BASE_DIGEST = real_builder.REQUIRED_BASE_DIGEST
                REQUIRED_IMAGE_TRT_VERSION = (
                    real_builder.REQUIRED_IMAGE_TRT_VERSION
                )
                REQUIRED_CUDA_VERSION = real_builder.REQUIRED_CUDA_VERSION
                REQUIRED_TRT_BANNER = real_builder.REQUIRED_TRT_BANNER
                RUNTIME_INPUT_CONTRACT = (
                    real_builder.RUNTIME_INPUT_CONTRACT
                )
                RUNTIME_ADAPTER_REVISION = (
                    real_builder.RUNTIME_ADAPTER_REVISION
                )
                RUNTIME_ADAPTER_SPEC = (
                    real_builder.RUNTIME_ADAPTER_SPEC
                )

                @staticmethod
                def _runtime_precision_profile(raw: str) -> object:
                    return real_builder._runtime_precision_profile(raw)

                @staticmethod
                def _runtime_engine_artifact_paths(
                    _root: Path,
                    _model: dict[str, object],
                    _profile: str,
                ) -> SimpleNamespace:
                    return paths

                @staticmethod
                def _validate_runtime_onnx_receipt(
                    _root: Path,
                    _paths: SimpleNamespace,
                    _model: dict[str, object],
                    _release: dict[str, object],
                ) -> dict[str, object]:
                    return runtime_source

                @staticmethod
                def _runtime_build_contract(
                    selected_model: dict[str, object],
                    selected_profile: str,
                    selected_precision: object,
                ) -> dict[str, object]:
                    return real_builder._runtime_build_contract(
                        selected_model,
                        selected_profile,
                        selected_precision,
                    )

                @staticmethod
                def _inner_build_command(
                    candidate_name: str,
                    *,
                    model: dict[str, object],
                    precision_profile: object | None,
                    layer_info_name: str | None,
                ) -> tuple[str, ...]:
                    return real_builder._inner_build_command(
                        candidate_name,
                        model=model,
                        precision_profile=precision_profile,
                        layer_info_name=layer_info_name,
                    )

                @staticmethod
                def _inner_load_command() -> tuple[str, ...]:
                    return real_builder._inner_load_command()

            evidence = root / "models/evidence"
            evidence.mkdir(parents=True)
            layer_info = evidence / "layer-info.json"
            layer_info.write_text('{"Layers":[{"Name":"head"}]}')
            commands = []
            command_values = [
                list(
                    real_builder._inner_build_command(
                        f"{paths.engine.name}.candidate",
                        model=model,
                        precision_profile=precision,
                        layer_info_name=(
                            f"{paths.engine.name}.layers.json"
                        ),
                    )
                ),
                list(real_builder._inner_load_command()),
                list(real_builder._inner_load_command()),
            ]
            for label, command in zip(
                ("build", "load-candidate", "load-installed"),
                command_values,
                strict=True,
            ):
                log = evidence / f"{label}.log"
                log.write_text(f"{label} passed")
                commands.append(
                    {
                        "label": label,
                        "command": command,
                        "returncode": 0,
                        "duration_seconds": 1.0,
                        "log": validator._relative(log, root),
                        "log_sha256": validator._sha256(log),
                        "status": "passed",
                    }
                )
            normalized = runtime_source["normalized_source"]
            receipt = {
                "schema": real_builder.RUNTIME_ENGINE_RECEIPT_SCHEMA,
                "promotion_status": "unpromoted",
                "runtime_selected": False,
                "artifact_role": "runtime_input_engine",
                "model_id": model["id"],
                "family": model["family"],
                "engine": {
                    "path": validator._relative(paths.engine, root),
                    "size_bytes": paths.engine.stat().st_size,
                    "sha256": validator._sha256(paths.engine),
                },
                "build_contract": real_builder._runtime_build_contract(
                    model, profile_id, precision
                ),
                "platform": {
                    "tensorrt_version": real_builder.REQUIRED_TRT_VERSION,
                    "image_id": real_builder.REQUIRED_IMAGE_ID,
                    "image_ref": real_builder.REQUIRED_IMAGE_REF,
                    "base_digest": real_builder.REQUIRED_BASE_DIGEST,
                    "image_tensorrt_version": (
                        real_builder.REQUIRED_IMAGE_TRT_VERSION
                    ),
                    "cuda_version": real_builder.REQUIRED_CUDA_VERSION,
                    "trtexec_banner": real_builder.REQUIRED_TRT_BANNER,
                    "expected_trtexec_banner": (
                        real_builder.REQUIRED_TRT_BANNER
                    ),
                },
                "source": {
                    "runtime_onnx": validator._relative(
                        paths.onnx, root
                    ),
                    "runtime_onnx_size_bytes": runtime_source[
                        "size_bytes"
                    ],
                    "runtime_onnx_sha256": runtime_source["sha256"],
                    "runtime_onnx_receipt": validator._relative(
                        paths.onnx_receipt, root
                    ),
                    "runtime_onnx_receipt_sha256": runtime_source[
                        "receipt_sha256"
                    ],
                    "tensor_contract": tensor_contract,
                },
                "runtime_input_contract": {
                    "input_contract": real_builder.RUNTIME_INPUT_CONTRACT,
                    "adapter_revision": (
                        real_builder.RUNTIME_ADAPTER_REVISION
                    ),
                    "adapter": real_builder.RUNTIME_ADAPTER_SPEC,
                    "normalized_source": {
                        "onnx": validator._relative(
                            normalized["path"], root
                        ),
                        "onnx_size_bytes": normalized["size_bytes"],
                        "onnx_sha256": normalized["sha256"],
                        "onnx_receipt": validator._relative(
                            normalized["receipt_path"], root
                        ),
                        "onnx_receipt_sha256": normalized[
                            "receipt_sha256"
                        ],
                    },
                },
                "commands": commands,
                "layer_info": {
                    "path": validator._relative(layer_info, root),
                    "size_bytes": layer_info.stat().st_size,
                    "sha256": validator._sha256(layer_info),
                },
            }
            receipt["receipt_sha256"] = validator._json_digest(receipt)
            paths.engine_receipt.parent.mkdir(parents=True, exist_ok=True)
            paths.engine_receipt.write_text(json.dumps(receipt))
            validated = validator._validate_engine_receipt(
                root,
                FakeBuilder,
                model,
                runtime_engine_profile=profile_id,
            )
            self.assertEqual(
                validated["sha256"], validator._sha256(paths.engine)
            )

            receipt["runtime_input_contract"][
                "adapter_revision"
            ] = "wrong"
            receipt["receipt_sha256"] = validator._json_digest(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }
            )
            paths.engine_receipt.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(
                validator.ValidationError,
                "runtime engine receipt contract",
            ):
                validator._validate_engine_receipt(
                    root,
                    FakeBuilder,
                    model,
                    runtime_engine_profile=profile_id,
                )

            baseline_profile_id = (
                validator.BASELINE_RUNTIME_ENGINE_PROFILE
            )
            baseline_precision = real_builder._runtime_precision_profile(
                baseline_profile_id
            )
            self.assertIsNone(baseline_precision)
            receipt["runtime_input_contract"][
                "adapter_revision"
            ] = real_builder.RUNTIME_ADAPTER_REVISION
            receipt["build_contract"] = real_builder._runtime_build_contract(
                model,
                baseline_profile_id,
                baseline_precision,
            )
            baseline_build_command = list(
                real_builder._inner_build_command(
                    f"{paths.engine.name}.candidate",
                    model=model,
                    precision_profile=baseline_precision,
                    layer_info_name=None,
                )
            )
            self.assertIn("--fp16", baseline_build_command)
            self.assertFalse(
                any(
                    "LayerInfo" in argument
                    or argument == "--dumpLayerInfo"
                    for argument in baseline_build_command
                )
            )
            receipt["commands"][0]["command"] = baseline_build_command
            receipt.pop("layer_info")
            receipt["receipt_sha256"] = validator._json_digest(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }
            )
            paths.engine_receipt.write_text(json.dumps(receipt))
            validated = validator._validate_engine_receipt(
                root,
                FakeBuilder,
                model,
                runtime_engine_profile=baseline_profile_id,
            )
            self.assertEqual(
                validated["sha256"], validator._sha256(paths.engine)
            )

            receipt["layer_info"] = {
                "path": validator._relative(layer_info, root),
                "size_bytes": layer_info.stat().st_size,
                "sha256": validator._sha256(layer_info),
            }
            receipt["receipt_sha256"] = validator._json_digest(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "receipt_sha256"
                }
            )
            paths.engine_receipt.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(
                validator.ValidationError,
                "runtime engine receipt contract",
            ):
                validator._validate_engine_receipt(
                    root,
                    FakeBuilder,
                    model,
                    runtime_engine_profile=baseline_profile_id,
                )

    def test_runner_manifest_requires_exact_tensor_and_output_contract(self) -> None:
        matrix = validator._load_matrix()
        model = next(
            row
            for row in matrix["models"]
            if row["id"] == "detect_nano"
        )
        manifest = _runner_manifest(model)
        validated = validator._validate_runner_manifest(
            manifest, model, "batch-180"
        )
        self.assertEqual(
            list(validated),
            ["dets", "labels"],
        )

        reordered_tensors = json.loads(json.dumps(manifest))
        reordered_tensors["tensors"][0], reordered_tensors["tensors"][1] = (
            reordered_tensors["tensors"][1],
            reordered_tensors["tensors"][0],
        )
        with self.assertRaises(validator.ValidationError):
            validator._validate_runner_manifest(
                reordered_tensors, model, "batch-180"
            )

        nonfinite_input = json.loads(json.dumps(manifest))
        nonfinite_input["input"]["min"] = float("nan")
        with self.assertRaises(validator.ValidationError):
            validator._validate_runner_manifest(
                nonfinite_input, model, "batch-180"
            )

        extra = json.loads(json.dumps(manifest))
        extra["outputs"].append(
            {
                "name": "surprise",
                "shape": [1],
                "byte_count": 4,
                "file": "batch-180__surprise.bin",
                "repeat_stable": True,
                "finite": True,
                "min": 0.0,
                "max": 0.0,
            }
        )
        with self.assertRaisesRegex(
            validator.ValidationError, "runner outputs"
        ):
            validator._validate_runner_manifest(extra, model, "batch-180")

        wrong_shape = json.loads(json.dumps(manifest))
        wrong_shape["outputs"][0]["shape"][1] = 299
        with self.assertRaisesRegex(
            validator.ValidationError, "runner output"
        ):
            validator._validate_runner_manifest(
                wrong_shape, model, "batch-180"
            )

        unsafe_file = json.loads(json.dumps(manifest))
        unsafe_file["outputs"][0]["file"] = "../dets.bin"
        with self.assertRaisesRegex(
            validator.ValidationError, "runner output"
        ):
            validator._validate_runner_manifest(
                unsafe_file, model, "batch-180"
            )

    def test_tokens_and_private_output_directories_fail_closed(self) -> None:
        self.assertEqual(
            validator._safe_token("batch-180", "case"), "batch-180"
        )
        self.assertEqual(
            validator._safe_filename("model_b3.onnx", "ONNX filename"),
            "model_b3.onnx",
        )
        for unsafe in ("", "../escape", "a/b", ".hidden", "évidence"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(validator.ValidationError):
                    validator._safe_token(unsafe, "case")
        for unsafe in ("", "../escape.onnx", "a/model.onnx", ".hidden"):
            with self.subTest(unsafe_filename=unsafe):
                with self.assertRaises(validator.ValidationError):
                    validator._safe_filename(unsafe, "ONNX filename")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private"
            root.mkdir(mode=0o700)
            self.assertEqual(
                validator._require_private_directory(root, "run root"), root
            )
            root.chmod(0o750)
            with self.assertRaisesRegex(
                validator.ValidationError, "0700"
            ):
                validator._require_private_directory(root, "run root")


if __name__ == "__main__":
    unittest.main()
