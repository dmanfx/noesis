from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


DS9_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = DS9_ROOT / "scripts" / "export_rfdetr_1_8_3.py"
SPEC = importlib.util.spec_from_file_location(
    "rfdetr_1_8_3_export_runtime_adapter", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def _identity_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="Body/Identity")],
        "runtime-adapter-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 2, 2])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="noesis-test",
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    return model


class RFDETR183RuntimeAdapterTests(unittest.TestCase):
    def test_matrix_declares_one_exact_runtime_contract_per_model(self) -> None:
        matrix = exporter._load_matrix()
        self.assertEqual(len(matrix["models"]), 13)
        runtime_names = []
        for model in matrix["models"]:
            runtime = model["runtime"]
            self.assertEqual(
                runtime["input_contract"], exporter.RUNTIME_INPUT_CONTRACT
            )
            self.assertEqual(
                runtime["adapter_revision"], exporter.RUNTIME_ADAPTER_REVISION
            )
            runtime_names.append(runtime["onnx_filename"])
        self.assertEqual(len(runtime_names), len(set(runtime_names)))

    def test_adapter_matches_exact_imagenet_normalization(self) -> None:
        adapted = exporter._inject_runtime_normalization(_identity_model())
        onnx.checker.check_model(adapted)
        node_names = [node.name for node in adapted.graph.node]
        self.assertEqual(
            node_names[:2],
            [exporter.RUNTIME_SUB_NODE, exporter.RUNTIME_DIV_NODE],
        )

        raw = np.asarray(
            [
                [
                    [[0.0, 0.25], [0.5, 1.0]],
                    [[0.1, 0.3], [0.6, 0.9]],
                    [[0.2, 0.4], [0.7, 0.8]],
                ]
            ],
            dtype=np.float32,
        )
        expected = (
            raw
            - np.asarray(exporter.RUNTIME_MEAN, dtype=np.float32).reshape(
                1, 3, 1, 1
            )
        ) / np.asarray(exporter.RUNTIME_STD, dtype=np.float32).reshape(
            1, 3, 1, 1
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapted.onnx"
            onnx.save_model(adapted, path)
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            observed = session.run(["output"], {"input": raw})[0]
        np.testing.assert_array_equal(observed, expected)

    def test_adapter_rejects_an_unused_public_input(self) -> None:
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["output"],
                    value=helper.make_tensor(
                        "constant", TensorProto.FLOAT, [1], [0.0]
                    ),
                )
            ],
            "unused-input",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        with self.assertRaisesRegex(RuntimeError, "does not consume"):
            exporter._inject_runtime_normalization(model)


if __name__ == "__main__":
    unittest.main()
