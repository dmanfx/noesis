from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "DS9" / "scripts" / "ds9_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "ds9_preflight_webrtc_dependency_test", PREFLIGHT_PATH
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


class DS9WebRTCDependencyPreflightTests(unittest.TestCase):
    def test_exact_gateway_namespaces_and_versions_are_initialized(self) -> None:
        require_version = mock.Mock()
        gst_init = mock.Mock()
        modules = {
            "gi": SimpleNamespace(require_version=require_version),
            "gi.repository.GLib": SimpleNamespace(),
            "gi.repository.Gst": SimpleNamespace(init=gst_init),
            "gi.repository.GstSdp": SimpleNamespace(),
            "gi.repository.GstWebRTC": SimpleNamespace(),
        }

        with mock.patch.object(
            preflight.importlib,
            "import_module",
            side_effect=lambda name: modules[name],
        ) as import_module:
            self.assertTrue(preflight._webrtc_python_stack_ok())

        self.assertEqual(
            require_version.call_args_list,
            [
                mock.call("Gst", "1.0"),
                mock.call("GstSdp", "1.0"),
                mock.call("GstWebRTC", "1.0"),
            ],
        )
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            [
                "gi",
                "gi.repository.GLib",
                "gi.repository.Gst",
                "gi.repository.GstSdp",
                "gi.repository.GstWebRTC",
            ],
        )
        gst_init.assert_called_once_with(None)

    def test_missing_pygobject_fails_closed_with_actionable_error(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                preflight.importlib,
                "import_module",
                side_effect=ModuleNotFoundError("No module named 'gi'"),
            ),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertFalse(preflight._webrtc_python_stack_ok())

        self.assertIn("DS9 WebRTC Python stack unavailable", stderr.getvalue())
        self.assertIn("PyGObject", stderr.getvalue())
        self.assertIn("No module named 'gi'", stderr.getvalue())

    def test_missing_required_namespace_fails_closed(self) -> None:
        gi = SimpleNamespace(require_version=mock.Mock())

        def import_module(name: str):
            if name == "gi":
                return gi
            if name == "gi.repository.GstWebRTC":
                raise ImportError("typelib GstWebRTC not found")
            if name == "gi.repository.Gst":
                return SimpleNamespace(init=mock.Mock())
            return SimpleNamespace()

        stderr = io.StringIO()
        with (
            mock.patch.object(
                preflight.importlib, "import_module", side_effect=import_module
            ),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertFalse(preflight._webrtc_python_stack_ok())

        self.assertIn("typelib GstWebRTC not found", stderr.getvalue())

    def test_python_module_preflight_propagates_webrtc_failure(self) -> None:
        def import_module(name: str):
            if name == "pyservicemaker":
                return SimpleNamespace(__file__="/ds9/pyservicemaker/__init__.py")
            if name == "pyds":
                raise ModuleNotFoundError("No module named 'pyds'")
            raise AssertionError(f"unexpected import: {name}")

        with (
            mock.patch.object(
                preflight.importlib, "import_module", side_effect=import_module
            ),
            mock.patch.object(
                preflight, "_webrtc_python_stack_ok", return_value=False
            ) as webrtc_check,
        ):
            self.assertFalse(preflight._python_modules_ok())

        webrtc_check.assert_called_once_with()

    def test_plugin_preflight_checks_every_gateway_factory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-webrtc-factories-") as raw:
            plugin_dir = Path(raw) / "gst-plugins"
            plugin_dir.mkdir()
            for binary_name in preflight.DS9_OWNED_GST_PLUGINS.values():
                (plugin_dir / binary_name).write_bytes(b"fixture")

            inspected: list[str] = []

            def fake_run(command, **_kwargs):
                element = str(command[-1])
                inspected.append(element)
                binary_name = preflight.DS9_OWNED_GST_PLUGINS.get(element)
                output = "Factory Details:\n"
                if binary_name is not None:
                    output += (
                        "\nPlugin Details:\n"
                        f"  Filename                 {plugin_dir / binary_name}\n"
                    )
                    if element == "nvdsroiexclude":
                        output += "\nElement Properties:\n"
                        output += "".join(
                            f"  {name:<20}: fixture\n"
                            for name in sorted(
                                preflight.NVDSROIEXCLUDE_REQUIRED_PROPERTIES
                            )
                        )
                return subprocess.CompletedProcess(command, 0, output, "")

            with (
                mock.patch.object(preflight, "DS9_GST_PLUGIN_DIR", plugin_dir),
                mock.patch.object(
                    preflight.shutil, "which", return_value="gst-inspect-1.0"
                ),
                mock.patch.object(preflight.subprocess, "run", side_effect=fake_run),
            ):
                self.assertTrue(preflight._plugins_ok())

        self.assertTrue(
            set(preflight.WEBRTC_GATEWAY_GST_FACTORIES).issubset(inspected)
        )

    def test_missing_webrtcbin_fails_plugin_preflight_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-webrtc-factories-") as raw:
            plugin_dir = Path(raw) / "gst-plugins"
            plugin_dir.mkdir()
            for binary_name in preflight.DS9_OWNED_GST_PLUGINS.values():
                (plugin_dir / binary_name).write_bytes(b"fixture")

            def fake_run(command, **_kwargs):
                element = str(command[-1])
                if element == "webrtcbin":
                    return subprocess.CompletedProcess(
                        command, 1, "", "No such element or plugin"
                    )
                binary_name = preflight.DS9_OWNED_GST_PLUGINS.get(element)
                output = "Factory Details:\n"
                if binary_name is not None:
                    output += (
                        "\nPlugin Details:\n"
                        f"  Filename                 {plugin_dir / binary_name}\n"
                    )
                    if element == "nvdsroiexclude":
                        output += "\nElement Properties:\n"
                        output += "".join(
                            f"  {name:<20}: fixture\n"
                            for name in sorted(
                                preflight.NVDSROIEXCLUDE_REQUIRED_PROPERTIES
                            )
                        )
                return subprocess.CompletedProcess(command, 0, output, "")

            stderr = io.StringIO()
            with (
                mock.patch.object(preflight, "DS9_GST_PLUGIN_DIR", plugin_dir),
                mock.patch.object(
                    preflight.shutil, "which", return_value="gst-inspect-1.0"
                ),
                mock.patch.object(preflight.subprocess, "run", side_effect=fake_run),
                contextlib.redirect_stderr(stderr),
            ):
                self.assertFalse(preflight._plugins_ok())

        self.assertIn("GStreamer element unavailable: webrtcbin", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
