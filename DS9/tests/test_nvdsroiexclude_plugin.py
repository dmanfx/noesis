from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DS8_PLUGIN_ROOT = ROOT / "gst-plugins"
DS9_PLUGIN_ROOT = ROOT / "DS9" / "gst-plugins"
DS8_SOURCE = DS8_PLUGIN_ROOT / "nvdsroiexclude" / "gstnvdsroiexclude.cpp"
DS9_SOURCE = ROOT / "DS9" / "csrc" / "nvdsroiexclude" / "gstnvdsroiexclude.cpp"


def _config(*, offset: int = 0, stream_suffix: str = "0") -> bytes:
    low = 2 + offset
    high = 30 + offset
    return textwrap.dedent(
        f"""
        [property]
        enable = 1
        osd-mode = 0
        display-font-size = 4
        config-width = 64
        config-height = 64

        [roi-filtering-stream-{stream_suffix}]
        enable = 1
        class-id = 0
        inverse-roi = 0
        config-width = 64
        config-height = 64
        roi-zone = {low};{low};{high};{low};{high};{high};{low};{high}
        enable-zone = 1
        """
    ).lstrip().encode("utf-8")


def _disabled_empty_config(*, stream_suffix: str = "0") -> bytes:
    return textwrap.dedent(
        f"""
        [property]
        enable = 1
        osd-mode = 0
        display-font-size = 4
        config-width = 64
        config-height = 64

        [roi-filtering-stream-{stream_suffix}]
        enable = 0
        class-id = 0
        inverse-roi = 0
        config-width = 64
        config-height = 64
        """
    ).lstrip().encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


GI_RELOAD_PROGRAM = textwrap.dedent(
    """
    import hashlib
    import json
    import os
    from pathlib import Path

    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    config_path = Path(os.environ["NOESIS_ROI_CONFIG"])
    next_path = Path(os.environ["NOESIS_ROI_NEXT_CONFIG"])
    invalid_path = Path(os.environ["NOESIS_ROI_INVALID_CONFIG"])
    expected_binary = Path(os.environ["NOESIS_ROI_PLUGIN_BINARY"]).resolve()

    pipeline = Gst.Pipeline.new("roi-reload-contract")
    source = Gst.ElementFactory.make("videotestsrc", "source")
    exclude = Gst.ElementFactory.make("nvdsroiexclude", "exclude")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    assert source is not None and exclude is not None and sink is not None
    actual_binary = Path(exclude.get_factory().get_plugin().get_filename()).resolve()
    assert actual_binary == expected_binary, (actual_binary, expected_binary)
    source.set_property("is-live", True)
    exclude.set_property("config-file", str(config_path))
    sink.set_property("sync", False)
    pipeline.add(source)
    pipeline.add(exclude)
    pipeline.add(sink)
    assert source.link(exclude)
    assert exclude.link(sink)

    initial_bytes = config_path.read_bytes()
    initial_sha = hashlib.sha256(initial_bytes).hexdigest()
    assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
    _, current, _ = pipeline.get_state(5 * Gst.SECOND)
    assert current == Gst.State.PLAYING, current

    initial = {
        "request": exclude.get_property("reload-request-sequence"),
        "accepted": exclude.get_property("reload-accepted-sequence"),
        "failed": exclude.get_property("reload-failed-sequence"),
        "ok": exclude.get_property("last-reload-ok"),
        "active": exclude.get_property("active-config-sha256"),
        "errors": exclude.get_property("reload-error-count"),
        "error": exclude.get_property("last-reload-error"),
    }

    next_bytes = next_path.read_bytes()
    next_sha = hashlib.sha256(next_bytes).hexdigest()
    config_path.write_bytes(next_bytes)
    exclude.set_property("expected-config-sha256", next_sha)
    exclude.set_property("reload-request-sequence", 7)
    accepted = {
        "request": exclude.get_property("reload-request-sequence"),
        "accepted": exclude.get_property("reload-accepted-sequence"),
        "failed": exclude.get_property("reload-failed-sequence"),
        "ok": exclude.get_property("last-reload-ok"),
        "active": exclude.get_property("active-config-sha256"),
        "errors": exclude.get_property("reload-error-count"),
        "error": exclude.get_property("last-reload-error"),
    }

    config_path.write_bytes(initial_bytes)
    exclude.set_property("expected-config-sha256", "0" * 64)
    exclude.set_property("reload-request-sequence", 8)
    bad_digest = {
        "request": exclude.get_property("reload-request-sequence"),
        "accepted": exclude.get_property("reload-accepted-sequence"),
        "failed": exclude.get_property("reload-failed-sequence"),
        "ok": exclude.get_property("last-reload-ok"),
        "active": exclude.get_property("active-config-sha256"),
        "errors": exclude.get_property("reload-error-count"),
        "error": exclude.get_property("last-reload-error"),
    }

    invalid_bytes = invalid_path.read_bytes()
    config_path.write_bytes(invalid_bytes)
    exclude.set_property(
        "expected-config-sha256", hashlib.sha256(invalid_bytes).hexdigest()
    )
    exclude.set_property("reload-request-sequence", 9)
    _, retained_state, _ = pipeline.get_state(0)
    invalid = {
        "request": exclude.get_property("reload-request-sequence"),
        "accepted": exclude.get_property("reload-accepted-sequence"),
        "failed": exclude.get_property("reload-failed-sequence"),
        "ok": exclude.get_property("last-reload-ok"),
        "active": exclude.get_property("active-config-sha256"),
        "errors": exclude.get_property("reload-error-count"),
        "error": exclude.get_property("last-reload-error"),
        "state": retained_state.value_nick,
    }

    pipeline.set_state(Gst.State.NULL)
    print(json.dumps({
        "initial_sha": initial_sha,
        "next_sha": next_sha,
        "initial": initial,
        "accepted": accepted,
        "bad_digest": bad_digest,
        "invalid": invalid,
    }, sort_keys=True))
    """
)


GI_STARTUP_REJECTION_PROGRAM = textwrap.dedent(
    """
    import json
    import os
    from pathlib import Path

    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    expected_binary = Path(os.environ["NOESIS_ROI_PLUGIN_BINARY"]).resolve()
    paths = json.loads(os.environ["NOESIS_ROI_INVALID_PATHS"])
    results = {}
    for name, raw_path in paths.items():
        pipeline = Gst.Pipeline.new("roi-startup-rejection")
        source = Gst.ElementFactory.make("videotestsrc", "source")
        exclude = Gst.ElementFactory.make("nvdsroiexclude", "exclude")
        sink = Gst.ElementFactory.make("fakesink", "sink")
        assert source is not None and exclude is not None and sink is not None
        actual_binary = Path(exclude.get_factory().get_plugin().get_filename()).resolve()
        assert actual_binary == expected_binary, (actual_binary, expected_binary)
        source.set_property("is-live", True)
        exclude.set_property("config-file", raw_path)
        sink.set_property("sync", False)
        pipeline.add(source)
        pipeline.add(exclude)
        pipeline.add(sink)
        assert source.link(exclude)
        assert exclude.link(sink)
        change = pipeline.set_state(Gst.State.PLAYING)
        _, current, pending = pipeline.get_state(2 * Gst.SECOND)
        results[name] = {
            "change": change.value_nick,
            "current": current.value_nick,
            "pending": pending.value_nick,
            "accepted": exclude.get_property("reload-accepted-sequence"),
            "ok": exclude.get_property("last-reload-ok"),
            "active": exclude.get_property("active-config-sha256"),
            "errors": exclude.get_property("reload-error-count"),
            "error": exclude.get_property("last-reload-error"),
        }
        pipeline.set_state(Gst.State.NULL)
    print(json.dumps(results, sort_keys=True))
    """
)


SERVICEMAKER_PROGRAM = textwrap.dedent(
    """
    import hashlib
    import json
    import os
    from pathlib import Path

    from pyservicemaker import Pipeline

    config_path = Path(os.environ["NOESIS_ROI_CONFIG"])
    next_path = Path(os.environ["NOESIS_ROI_NEXT_CONFIG"])
    next_bytes = next_path.read_bytes()
    next_sha = hashlib.sha256(next_bytes).hexdigest()

    pipeline = Pipeline("roi-exclude-node-contract")
    pipeline.add("videotestsrc", "source", {"is-live": True})
    pipeline.add(
        "nvdsroiexclude",
        "exclude",
        {"config-file": str(config_path), "id-mode": "source-id"},
    )
    pipeline.add("fakesink", "sink", {"sync": False})
    pipeline.link("source", "exclude", "sink")
    pipeline.prepare()
    node = pipeline["exclude"]
    before = [
        node.get("reload-request-sequence"),
        node.get("reload-accepted-sequence"),
        node.get("last-reload-ok"),
        node.get("active-config-sha256"),
    ]
    pipeline.activate()
    config_path.write_bytes(next_bytes)
    node.set({"expected-config-sha256": next_sha})
    node.set({"reload-request-sequence": 31})
    after = [
        node.get("reload-request-sequence"),
        node.get("reload-accepted-sequence"),
        node.get("reload-failed-sequence"),
        node.get("last-reload-ok"),
        node.get("active-config-sha256"),
        node.get("last-reload-error"),
    ]
    pipeline.stop()
    pipeline.wait()
    print(json.dumps({"before": before, "after": after, "sha": next_sha}, sort_keys=True))
    """
)


class RoiExcludePluginSourceTests(unittest.TestCase):
    def test_ds8_and_ds9_sources_are_owned_mirrors(self) -> None:
        self.assertFalse(DS8_SOURCE.is_symlink())
        self.assertFalse(DS9_SOURCE.is_symlink())
        self.assertEqual(DS8_SOURCE.read_bytes(), DS9_SOURCE.read_bytes())

        source = DS8_SOURCE.read_text(encoding="utf-8")
        self.assertIn("nvds_remove_obj_meta_from_frame", source)
        self.assertIn("O_NOFOLLOW", source)
        self.assertIn("kMaxConfigBytes", source)
        self.assertIn("validate_unique_ini_keys", source)
        self.assertIn('"expected-config-sha256"', source)
        self.assertIn('"active-config-sha256"', source)
        self.assertIn('"reload-accepted-sequence"', source)
        self.assertIn("stream.enabled && stream.polygons.empty()", source)
        self.assertIn("configured analytics stream coverage drifted at runtime", source)
        self.assertNotIn("gst_buffer_map", source)
        self.assertNotIn("gst_buffer_make_writable", source)

    def test_builds_are_sdk_major_scoped_and_fail_closed(self) -> None:
        ds8_cmake = (DS8_SOURCE.parent / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        ds9_cmake = (DS9_SOURCE.parent / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn('NOESIS_DEEPSTREAM_MAJOR="8"', ds8_cmake)
        self.assertNotIn('NOESIS_DEEPSTREAM_MAJOR="9"', ds8_cmake)
        self.assertIn('NVDS_VERSION_MAJOR[ \\t]+8', ds8_cmake)
        self.assertIn("-Werror", ds8_cmake)
        self.assertIn('NOESIS_DEEPSTREAM_MAJOR="9"', ds9_cmake)
        self.assertNotIn('NOESIS_DEEPSTREAM_MAJOR="8"', ds9_cmake)
        self.assertIn('NVDS_VERSION_MAJOR[ \\t]+9', ds9_cmake)
        self.assertIn("-Werror", ds9_cmake)

        ds8_build = (DS8_PLUGIN_ROOT / "build_nvdsroiexclude.sh").read_text(
            encoding="utf-8"
        )
        ds9_build = (ROOT / "DS9" / "scripts" / "build_gst_plugins.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('NVDS_VERSION_MAJOR[[:space:]]+8', ds8_build)
        self.assertIn('-DDEEPSTREAM_HOME="${DS_HOME}"', ds8_build)
        self.assertIn('-DDEEPSTREAM_HOME="${DS_HOME}"', ds9_build)

    def test_cmake_rejects_cross_major_sdk_roots(self) -> None:
        ds8_home = Path("/opt/nvidia/deepstream/deepstream-8.0")
        ds9_home = Path("/opt/nvidia/deepstream/deepstream-9.0")
        if not ds8_home.is_dir() or not ds9_home.is_dir():
            self.skipTest("both DS8 and DS9 SDK roots are required for cross-major guard")

        cases = (
            (DS8_SOURCE.parent, ds9_home, "DeepStream major 8"),
            (DS9_SOURCE.parent, ds8_home, "DeepStream major 9"),
        )
        with tempfile.TemporaryDirectory(prefix="noesis-roi-cmake-guard-") as raw:
            root = Path(raw)
            for index, (source_dir, wrong_home, expected) in enumerate(cases):
                with self.subTest(source_dir=source_dir, wrong_home=wrong_home):
                    result = subprocess.run(
                        [
                            "cmake",
                            "--fresh",
                            "-S",
                            str(source_dir),
                            "-B",
                            str(root / f"build-{index}"),
                            f"-DDEEPSTREAM_HOME={wrong_home}",
                        ],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stdout + result.stderr)


class RoiExcludePluginRuntimeTests(unittest.TestCase):
    def _plugin_binary(self, plugin_root: Path) -> Path:
        binary = plugin_root / "libgstnvdsroiexclude.so"
        if not binary.is_file():
            self.skipTest(f"plugin has not been built: {binary}")
        return binary

    def _run(
        self,
        program: str,
        *,
        plugin_root: Path = DS8_PLUGIN_ROOT,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        binary = self._plugin_binary(plugin_root)
        with tempfile.TemporaryDirectory(prefix="noesis-roi-registry-") as raw:
            env = os.environ.copy()
            env["GST_PLUGIN_PATH"] = str(plugin_root)
            env["GST_REGISTRY"] = str(Path(raw) / "registry.bin")
            env["NOESIS_ROI_PLUGIN_BINARY"] = str(binary)
            if extra_env:
                env.update(extra_env)
            return subprocess.run(
                [sys.executable, "-c", program],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
            )

    def test_ds8_and_ds9_binaries_register_reload_contract(self) -> None:
        properties = (
            "config-file",
            "id-mode",
            "reload-request-sequence",
            "reload-accepted-sequence",
            "reload-failed-sequence",
            "last-reload-ok",
            "expected-config-sha256",
            "active-config-sha256",
            "reload-error-count",
            "objects-removed-count",
            "last-reload-error",
        )
        for plugin_root in (DS8_PLUGIN_ROOT, DS9_PLUGIN_ROOT):
            with self.subTest(plugin_root=plugin_root):
                binary = self._plugin_binary(plugin_root)
                with tempfile.TemporaryDirectory(prefix="noesis-roi-inspect-") as raw:
                    env = os.environ.copy()
                    env["GST_PLUGIN_PATH"] = str(plugin_root)
                    env["GST_REGISTRY"] = str(Path(raw) / "registry.bin")
                    result = subprocess.run(
                        ["gst-inspect-1.0", "nvdsroiexclude"],
                        cwd=ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(str(binary), result.stdout)
                for prop in properties:
                    self.assertIn(prop, result.stdout)
                self.assertIn("Capabilities:\n      ANY", result.stdout)

    def _assert_gi_reload_contract(self, plugin_root: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="noesis-roi-reload-") as raw:
            root = Path(raw)
            config_path = root / "active.ini"
            next_path = root / "next.ini"
            invalid_path = root / "invalid.ini"
            initial = _config(offset=0)
            next_config = _config(offset=5)
            config_path.write_bytes(initial)
            next_path.write_bytes(next_config)
            invalid_path.write_bytes(
                _config().replace(b"30;2", b"nan;2", 1)
            )
            result = self._run(
                GI_RELOAD_PROGRAM,
                plugin_root=plugin_root,
                extra_env={
                    "NOESIS_ROI_CONFIG": str(config_path),
                    "NOESIS_ROI_NEXT_CONFIG": str(next_path),
                    "NOESIS_ROI_INVALID_CONFIG": str(invalid_path),
                },
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        initial_sha = _sha256(initial)
        next_sha = _sha256(next_config)
        self.assertEqual(payload["initial_sha"], initial_sha)
        self.assertEqual(payload["next_sha"], next_sha)
        self.assertEqual(
            payload["initial"],
            {
                "accepted": 0,
                "active": initial_sha,
                "error": "",
                "errors": 0,
                "failed": 0,
                "ok": True,
                "request": 0,
            },
        )
        self.assertEqual(
            payload["accepted"],
            {
                "accepted": 7,
                "active": next_sha,
                "error": "",
                "errors": 0,
                "failed": 0,
                "ok": True,
                "request": 7,
            },
        )
        self.assertEqual(payload["bad_digest"]["accepted"], 7)
        self.assertEqual(payload["bad_digest"]["active"], next_sha)
        self.assertEqual(payload["bad_digest"]["failed"], 8)
        self.assertEqual(payload["bad_digest"]["errors"], 1)
        self.assertFalse(payload["bad_digest"]["ok"])
        self.assertIn("does not match", payload["bad_digest"]["error"])
        self.assertEqual(payload["invalid"]["accepted"], 7)
        self.assertEqual(payload["invalid"]["active"], next_sha)
        self.assertEqual(payload["invalid"]["failed"], 9)
        self.assertEqual(payload["invalid"]["errors"], 2)
        self.assertFalse(payload["invalid"]["ok"])
        self.assertEqual(payload["invalid"]["state"], "playing")
        self.assertIn("invalid ROI coordinate", payload["invalid"]["error"])

    def test_gi_reload_acknowledges_exact_hash_and_retains_prior_on_failure(self) -> None:
        for plugin_root in (DS8_PLUGIN_ROOT, DS9_PLUGIN_ROOT):
            with self.subTest(plugin_root=plugin_root):
                self._assert_gi_reload_contract(plugin_root)

    def test_disabled_empty_stream_is_a_valid_reload_policy(self) -> None:
        for plugin_root in (DS8_PLUGIN_ROOT, DS9_PLUGIN_ROOT):
            with self.subTest(plugin_root=plugin_root):
                with tempfile.TemporaryDirectory(prefix="noesis-roi-disabled-empty-") as raw:
                    root = Path(raw)
                    config_path = root / "active.ini"
                    next_path = root / "next.ini"
                    invalid_path = root / "invalid.ini"
                    config_path.write_bytes(_config())
                    disabled = _disabled_empty_config()
                    next_path.write_bytes(disabled)
                    invalid_path.write_bytes(
                        _config().replace(b"30;2", b"nan;2", 1)
                    )
                    result = self._run(
                        GI_RELOAD_PROGRAM,
                        plugin_root=plugin_root,
                        extra_env={
                            "NOESIS_ROI_CONFIG": str(config_path),
                            "NOESIS_ROI_NEXT_CONFIG": str(next_path),
                            "NOESIS_ROI_INVALID_CONFIG": str(invalid_path),
                        },
                    )

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                payload = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(payload["accepted"]["accepted"], 7)
                self.assertEqual(payload["accepted"]["active"], _sha256(disabled))
                self.assertTrue(payload["accepted"]["ok"])

    def test_unsafe_or_ambiguous_configs_fail_at_startup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="noesis-roi-invalid-") as raw:
            root = Path(raw)
            valid_path = root / "valid.ini"
            valid_path.write_bytes(_config())

            invalid: dict[str, Path] = {}

            symlink = root / "symlink.ini"
            symlink.symlink_to(valid_path)
            invalid["symlink"] = symlink

            oversize = root / "oversize.ini"
            oversize.write_bytes(b"x" * (1024 * 1024 + 1))
            invalid["oversize"] = oversize

            variants = {
                "nonfinite": _config().replace(b"30;2", b"nan;2", 1),
                "out_of_bounds": _config().replace(b"30;2", b"65;2", 1),
                "trailing_stream_suffix": _config(stream_suffix="0junk"),
                "duplicate_group": _config()
                + b"\n[roi-filtering-stream-0]\nroi-other = 1;1;2;1;2;2\n",
                "duplicate_key": _config().replace(
                    b"enable = 1\n", b"enable = 1\nenable = 1\n", 1
                ),
                "canonical_stream_collision": _config()
                + _config(offset=3, stream_suffix="00").split(b"\n\n", 1)[1],
                "collinear": _config().replace(
                    b"2;2;30;2;30;30;2;30", b"2;2;10;10;20;20;30;30"
                ),
            }
            for name, payload in variants.items():
                path = root / f"{name}.ini"
                path.write_bytes(payload)
                invalid[name] = path

            missing = root / "missing.ini"
            invalid["missing"] = missing

            result = self._run(
                GI_STARTUP_REJECTION_PROGRAM,
                extra_env={
                    "NOESIS_ROI_INVALID_PATHS": json.dumps(
                        {name: str(path) for name, path in invalid.items()},
                        sort_keys=True,
                    )
                },
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(set(payload), set(invalid))
        for name, status in payload.items():
            with self.subTest(name=name):
                self.assertNotEqual(status["current"], "playing")
                self.assertFalse(status["ok"])
                self.assertEqual(status["accepted"], 0)
                self.assertEqual(status["active"], "")
                self.assertGreaterEqual(status["errors"], 1)
                self.assertTrue(status["error"])
                self.assertNotEqual(status["error"], "not_loaded")

    def test_servicemaker_node_round_trips_reload_ack_and_hash(self) -> None:
        try:
            import pyservicemaker  # noqa: F401
        except ImportError:
            self.skipTest("pyservicemaker is unavailable")

        with tempfile.TemporaryDirectory(prefix="noesis-roi-node-") as raw:
            root = Path(raw)
            config_path = root / "active.ini"
            next_path = root / "next.ini"
            config_path.write_bytes(_config())
            next_config = _config(offset=7)
            next_path.write_bytes(next_config)
            result = self._run(
                SERVICEMAKER_PROGRAM,
                extra_env={
                    "NOESIS_ROI_CONFIG": str(config_path),
                    "NOESIS_ROI_NEXT_CONFIG": str(next_path),
                },
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload_line = next(
            line for line in reversed(result.stdout.splitlines()) if line.startswith("{")
        )
        payload = json.loads(payload_line)
        next_sha = _sha256(next_config)
        self.assertEqual(payload["sha"], next_sha)
        self.assertEqual(payload["before"], [0, 0, True, _sha256(_config())])
        self.assertEqual(payload["after"], [31, 31, 0, True, next_sha, ""])


if __name__ == "__main__":
    unittest.main()
