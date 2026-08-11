from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_RUNTIME_CORE = REPO_ROOT / "DS9" / "noesis" / "ds9_runtime_core.py"


class IdentityTopologyParityTests(unittest.TestCase):
    @staticmethod
    def _isolated_builder():
        source = DS9_RUNTIME_CORE.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(DS9_RUNTIME_CORE))
        builder = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_build_stable_id_manager"
        )
        module = ast.Module(body=[builder], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "Any": Any,
            "Dict": Dict,
            "Mapping": Mapping,
            "Optional": Optional,
            "Path": Path,
            "REPO_ROOT": REPO_ROOT,
            "inspect": inspect,
            "logging": logging,
            "os": os,
        }
        exec(compile(module, str(DS9_RUNTIME_CORE), "exec"), namespace)
        return namespace["_build_stable_id_manager"]

    def test_ds9_stable_id_builder_receives_active_camera_labels(self) -> None:
        tree = ast.parse(
            DS9_RUNTIME_CORE.read_text(encoding="utf-8"), filename=str(DS9_RUNTIME_CORE)
        )
        builder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_build_stable_id_manager"
        )
        self.assertIn("camera_labels", [arg.arg for arg in builder.args.kwonlyargs])

        runtime_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_stable_id_manager"
        ]
        self.assertEqual(len(runtime_calls), 1)
        camera_keyword = next(
            (
                keyword
                for keyword in runtime_calls[0].keywords
                if keyword.arg == "camera_labels"
            ),
            None,
        )
        self.assertIsNotNone(camera_keyword)
        assert camera_keyword is not None
        self.assertIsInstance(camera_keyword.value, ast.Name)
        self.assertEqual(camera_keyword.value.id, "camera_labels")

        builder_source = (
            ast.get_source_segment(
                DS9_RUNTIME_CORE.read_text(encoding="utf-8"), builder
            )
            or ""
        )
        self.assertIn(
            'extra_kwargs["camera_labels"] = dict(camera_labels)', builder_source
        )

    def test_ds9_household_builder_disables_legacy_auto_merge_fail_closed(self) -> None:
        builder = self._isolated_builder()
        created = []
        prepared = []

        class FakeStableIDManager:
            force_auto_merge = False

            def __init__(
                self,
                *,
                household_mode=False,
                auto_merge_enabled=True,
                allow_multi_zone_active=True,
                total_id_reuse=True,
                **kwargs,
            ) -> None:
                self.household_mode = bool(household_mode)
                self.auto_merge_enabled = bool(
                    self.force_auto_merge or auto_merge_enabled
                )
                self.allow_multi_zone_active = bool(allow_multi_zone_active)
                self.total_id_reuse = bool(total_id_reuse)
                self.kwargs = dict(kwargs)
                created.append(self)

            @staticmethod
            def get_sid_metrics():
                return {"stableid_backend_mode": "gpu"}

        stable_module = types.ModuleType("reid.stable_id_manager")
        stable_module.StableIDManager = FakeStableIDManager
        household_module = types.ModuleType("reid.household_state")
        household_module.is_household_identity_enabled = lambda: True

        def prepare(_logger, **kwargs):
            prepared.append(kwargs)
            return {
                "household_mode": True,
                "auto_merge_enabled": False,
                "allow_multi_zone_active": False,
                "total_id_reuse": False,
            }

        household_module.prepare_household_stable_id_overrides = prepare
        environment = {
            "NOESIS_REID_ENABLED": "1",
            "NOESIS_HOUSEHOLD_IDENTITY": "1",
            # An old deployment knob must never override household policy.
            "NOESIS_REID_AUTO_MERGE_ENABLED": "1",
            "NOESIS_STABLEID_GPU_ENABLED": "1",
        }
        with patch.dict(
            sys.modules,
            {
                "reid.stable_id_manager": stable_module,
                "reid.household_state": household_module,
            },
        ), patch.dict(os.environ, environment, clear=False):
            manager = builder(
                logging.getLogger("ds9-household-policy-test"),
                pipeline_config={"models": {"pose": {"enable": True}}},
                camera_labels={0: "camera-a"},
            )
            FakeStableIDManager.force_auto_merge = True
            rejected = builder(
                logging.getLogger("ds9-household-policy-rejection-test"),
                pipeline_config={"models": {"pose": {"enable": True}}},
                camera_labels={0: "camera-a"},
            )

        self.assertIsNotNone(manager)
        self.assertIsNone(rejected)
        self.assertEqual(len(created), 2)
        self.assertEqual(len(prepared), 2)
        self.assertTrue(created[0].household_mode)
        self.assertFalse(created[0].auto_merge_enabled)
        self.assertFalse(created[0].allow_multi_zone_active)
        self.assertFalse(created[0].total_id_reuse)


if __name__ == "__main__":
    unittest.main()
