from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT_PATH = REPO_ROOT / "DS9" / "scripts" / "ds9_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "ds9_preflight_plugin_origin_test", PREFLIGHT_PATH
)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def _inspect_output(filename: Path, *, properties: tuple[str, ...] = ()) -> str:
    output = (
        "Factory Details:\n"
        "  Long-name                fixture\n\n"
        "Plugin Details:\n"
        "  Name                     fixture\n"
        f"  Filename                 {filename}\n"
    )
    if properties:
        output += "\nElement Properties:\n"
        output += "".join(f"  {name:<20}: fixture\n" for name in properties)
    return output


class DS9PluginOriginPreflightTests(unittest.TestCase):
    def test_origin_parser_accepts_only_the_exact_ds9_owned_binary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-plugin-origin-") as raw:
            plugin_dir = Path(raw) / "DS9" / "gst-plugins"
            plugin_dir.mkdir(parents=True)
            expected = plugin_dir / "libgstnoesiseos.so"
            expected.write_bytes(b"ds9")

            actual = preflight._validate_owned_plugin_origin(
                "noesiseos", _inspect_output(expected), expected
            )

            self.assertEqual(actual, expected.resolve())

    def test_origin_parser_rejects_ds8_or_unreported_origins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-plugin-origin-") as raw:
            root = Path(raw)
            expected = root / "DS9" / "gst-plugins" / "libgstnoesiseos.so"
            ds8_binary = root / "gst-plugins" / "libgstnoesiseos.so"

            with self.assertRaisesRegex(ValueError, "required DS9-owned binary"):
                preflight._validate_owned_plugin_origin(
                    "noesiseos", _inspect_output(ds8_binary), expected
                )
            with self.assertRaisesRegex(ValueError, "did not report"):
                preflight._validate_owned_plugin_origin(
                    "noesiseos", "Plugin Details:\n", expected
                )

    def test_roi_contract_requires_every_exact_ack_hash_error_property(self) -> None:
        complete = tuple(sorted(preflight.NVDSROIEXCLUDE_REQUIRED_PROPERTIES))
        preflight._validate_nvdsroiexclude_contract(
            _inspect_output(Path("/tmp/libgstnvdsroiexclude.so"), properties=complete)
        )

        stale = tuple(
            name for name in complete if name != "reload-accepted-sequence"
        )
        with self.assertRaisesRegex(
            ValueError, "missing exact properties: reload-accepted-sequence"
        ):
            preflight._validate_nvdsroiexclude_contract(
                _inspect_output(
                    Path("/tmp/libgstnvdsroiexclude.so"), properties=stale
                )
            )

        with self.assertRaisesRegex(ValueError, "active-config-sha256"):
            preflight._validate_nvdsroiexclude_contract(
                "Element Properties:\n"
                "  description         : mentions active-config-sha256 only\n"
            )

    def test_plugin_preflight_validates_all_repo_owned_origins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-plugin-preflight-") as raw:
            plugin_dir = Path(raw) / "gst-plugins"
            plugin_dir.mkdir()
            for binary_name in preflight.DS9_OWNED_GST_PLUGINS.values():
                (plugin_dir / binary_name).write_bytes(b"fixture")

            inspected: list[str] = []

            def fake_run(command, **_kwargs):
                element = str(command[-1])
                inspected.append(element)
                binary_name = preflight.DS9_OWNED_GST_PLUGINS.get(element)
                output = (
                    _inspect_output(
                        plugin_dir / binary_name,
                        properties=(
                            tuple(sorted(preflight.NVDSROIEXCLUDE_REQUIRED_PROPERTIES))
                            if element == "nvdsroiexclude"
                            else ()
                        ),
                    )
                    if binary_name is not None
                    else "Factory Details:\n"
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

            self.assertTrue(set(preflight.DS9_OWNED_GST_PLUGINS).issubset(inspected))


if __name__ == "__main__":
    unittest.main()
