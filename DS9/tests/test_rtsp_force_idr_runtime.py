from __future__ import annotations

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

from noesis.ds8_runtime import _build_mosaic_keyframe_requester


ROOT = Path(__file__).resolve().parents[2]


def _function_ast(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.dump(node, include_attributes=False)


class _Trigger:
    def __init__(self, reject: bool = False) -> None:
        self.reject = reject
        self.accepted = 0
        self.last_ok = False
        self.requests: list[int] = []

    def get(self, name: str):
        if name == "accepted-sequence":
            return self.accepted
        if name == "last-request-ok":
            return self.last_ok
        raise KeyError(name)

    def set(self, values: dict[str, object]) -> None:
        requested = int(values["request-sequence"])
        self.requests.append(requested)
        if not self.reject:
            self.accepted = requested
            self.last_ok = True


class _ServiceMakerPipeline:
    def __init__(self, trigger: _Trigger) -> None:
        self.trigger = trigger

    def __getitem__(self, name: str) -> _Trigger:
        if name != "mosaic_force_idr":
            raise KeyError(name)
        return self.trigger


def _pipeline(trigger: _Trigger):
    return SimpleNamespace(
        ds_pipeline=_ServiceMakerPipeline(trigger),
        components={
            "mosaic_force_idr": SimpleNamespace(element="noesisforceidr")
        },
    )


class ForceIdrRuntimeTests(unittest.TestCase):
    def test_ds9_keyframe_requester_is_exact_ds8_parity(self) -> None:
        name = "_build_mosaic_keyframe_requester"
        self.assertEqual(
            _function_ast(ROOT / "noesis" / "ds8_runtime.py", name),
            _function_ast(ROOT / "DS9" / "noesis" / "ds9_runtime_core.py", name),
        )

    def test_monotonic_native_requests_are_acknowledged(self) -> None:
        trigger = _Trigger()
        requester = _build_mosaic_keyframe_requester(
            _pipeline(trigger), logging.getLogger("test")
        )
        self.assertIsNotNone(requester)
        assert requester is not None

        requester("webrtc_offer")
        requester("webrtc_connected")

        self.assertEqual(trigger.requests, [1, 2])
        self.assertEqual(trigger.accepted, 2)

    def test_rejected_native_request_is_reported(self) -> None:
        trigger = _Trigger(reject=True)
        failures: list[BaseException] = []
        requester = _build_mosaic_keyframe_requester(
            _pipeline(trigger),
            logging.getLogger("test"),
            failure_callback=failures.append,
        )
        self.assertIsNotNone(requester)
        assert requester is not None

        with self.assertRaises(RuntimeError):
            requester("webrtc_offer")
        self.assertEqual(len(failures), 1)


if __name__ == "__main__":
    unittest.main()
