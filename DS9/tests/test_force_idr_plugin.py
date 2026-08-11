from __future__ import annotations

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
DS8_SOURCE = DS8_PLUGIN_ROOT / "noesisforceidr" / "gstnoesisforceidr.cpp"
DS9_SOURCE = DS9_PLUGIN_ROOT / "noesisforceidr" / "gstnoesisforceidr.cpp"


class ForceIdrPluginSourceTests(unittest.TestCase):
    def test_source_uses_official_downstream_event_without_mapping_buffers(self) -> None:
        source = DS8_SOURCE.read_text(encoding="utf-8")

        self.assertIn("gst_nvevent_enc_force_idr", source)
        self.assertIn("gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(self), event)", source)
        self.assertIn('"request-sequence"', source)
        self.assertIn('"accepted-sequence"', source)
        self.assertIn('"last-request-ok"', source)
        self.assertIn('"stream-id"', source)
        self.assertIn("GST_STATIC_CAPS_ANY", source)
        self.assertIn("gst_base_transform_set_passthrough", source)
        self.assertIn("g_param_spec_uint(", source)
        self.assertNotIn("g_param_spec_uint64", source)
        self.assertNotIn("gst_buffer_map", source)
        self.assertNotIn("gst_buffer_make_writable", source)

    def test_ds8_and_ds9_sources_are_owned_mirrors(self) -> None:
        self.assertFalse(DS8_SOURCE.is_symlink())
        self.assertFalse(DS9_SOURCE.is_symlink())
        self.assertEqual(DS8_SOURCE.read_bytes(), DS9_SOURCE.read_bytes())

        ds8_cmake = (DS8_SOURCE.parent / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        ds9_cmake = (DS9_SOURCE.parent / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("deepstream-8.0", ds8_cmake)
        self.assertNotIn("deepstream-9.0", ds8_cmake)
        self.assertIn("deepstream-9.0", ds9_cmake)
        self.assertNotIn("deepstream-8.0", ds9_cmake)


class ForceIdrPluginRegistrationTests(unittest.TestCase):
    def _inspect(self, plugin_root: Path) -> str:
        binary = plugin_root / "libgstnoesisforceidr.so"
        if not binary.is_file():
            self.skipTest(f"plugin has not been built: {binary}")

        with tempfile.TemporaryDirectory(prefix="noesis-force-idr-test-") as raw:
            env = os.environ.copy()
            env["GST_PLUGIN_PATH"] = str(plugin_root)
            env["GST_REGISTRY"] = str(Path(raw) / "registry.bin")
            result = subprocess.run(
                ["gst-inspect-1.0", "noesisforceidr"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_ds8_and_ds9_binaries_register_the_strict_property_contract(self) -> None:
        for plugin_root in (DS8_PLUGIN_ROOT, DS9_PLUGIN_ROOT):
            with self.subTest(plugin_root=plugin_root):
                output = self._inspect(plugin_root)
                self.assertIn(str(plugin_root / "libgstnoesisforceidr.so"), output)
                self.assertIn("request-sequence", output)
                self.assertIn("accepted-sequence", output)
                self.assertIn("last-request-ok", output)
                self.assertIn("stream-id", output)
                self.assertIn("Capabilities:\n      ANY", output)

    def test_monotonic_request_acknowledges_exactly_one_official_event(self) -> None:
        binary = DS8_PLUGIN_ROOT / "libgstnoesisforceidr.so"
        if not binary.is_file():
            self.skipTest(f"plugin has not been built: {binary}")

        program = textwrap.dedent(
            """
            import json
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            pipeline = Gst.parse_launch(
                "appsrc name=source is-live=true format=time "
                "caps=video/x-raw,format=I420,width=16,height=16,framerate=1/1 "
                "! noesisforceidr name=trigger stream-id=mosaic "
                "! fakesink name=sink sync=false"
            )
            trigger = pipeline.get_by_name("trigger")
            sink = pipeline.get_by_name("sink")
            seen = []

            def probe(_pad, info):
                event = info.get_event()
                structure = event.get_structure() if event is not None else None
                if structure is not None and structure.get_name() == "nv-enc-force-idr":
                    seen.append(
                        [structure.get_string("stream_id"), structure.get_value("force")]
                    )
                return Gst.PadProbeReturn.OK

            sink.get_static_pad("sink").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, probe)
            assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
            source = pipeline.get_by_name("source")
            buffer = Gst.Buffer.new_allocate(None, 16 * 16 * 3 // 2, None)
            buffer.pts = 0
            buffer.duration = Gst.SECOND
            assert source.emit("push-buffer", buffer) == Gst.FlowReturn.OK
            pipeline.get_state(2 * Gst.SECOND)

            trigger.set_property("request-sequence", 7)
            trigger.set_property("request-sequence", 7)
            trigger.set_property("request-sequence", 6)
            payload = {
                "request": trigger.get_property("request-sequence"),
                "accepted": trigger.get_property("accepted-sequence"),
                "ok": trigger.get_property("last-request-ok"),
                "events": seen,
            }
            pipeline.set_state(Gst.State.NULL)
            print(json.dumps(payload, sort_keys=True))
            """
        )

        with tempfile.TemporaryDirectory(prefix="noesis-force-idr-event-") as raw:
            env = os.environ.copy()
            env["GST_PLUGIN_PATH"] = str(DS8_PLUGIN_ROOT)
            env["GST_REGISTRY"] = str(Path(raw) / "registry.bin")
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(
            payload,
            {
                "accepted": 7,
                "events": [["mosaic", 1]],
                "ok": True,
                "request": 7,
            },
        )

    def test_servicemaker_node_round_trips_uint_sequence_and_ack(self) -> None:
        binary = DS8_PLUGIN_ROOT / "libgstnoesisforceidr.so"
        if not binary.is_file():
            self.skipTest(f"plugin has not been built: {binary}")
        try:
            import pyservicemaker  # noqa: F401
        except ImportError:
            self.skipTest("pyservicemaker is unavailable")

        program = textwrap.dedent(
            """
            import json
            from pyservicemaker import Pipeline

            pipeline = Pipeline("force-idr-node-contract")
            pipeline.add("videotestsrc", "source", {"is-live": True})
            pipeline.add("noesisforceidr", "trigger", {"stream-id": "mosaic"})
            pipeline.add("fakesink", "sink", {"sync": False})
            pipeline.link("source", "trigger", "sink")
            pipeline.prepare()
            node = pipeline["trigger"]
            before = [
                node.get("request-sequence"),
                node.get("accepted-sequence"),
                node.get("last-request-ok"),
            ]
            pipeline.activate()
            node.set({"request-sequence": 23})
            after = [
                node.get("request-sequence"),
                node.get("accepted-sequence"),
                node.get("last-request-ok"),
                node.get("stream-id"),
            ]
            pipeline.stop()
            pipeline.wait()
            print(json.dumps({"before": before, "after": after}, sort_keys=True))
            """
        )

        with tempfile.TemporaryDirectory(prefix="noesis-force-idr-node-") as raw:
            env = os.environ.copy()
            env["GST_PLUGIN_PATH"] = str(DS8_PLUGIN_ROOT)
            env["GST_REGISTRY"] = str(Path(raw) / "registry.bin")
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload_line = next(
            line for line in reversed(result.stdout.splitlines()) if line.startswith("{")
        )
        self.assertEqual(
            json.loads(payload_line),
            {"after": [23, 23, True, "mosaic"], "before": [0, 0, False]},
        )


if __name__ == "__main__":
    unittest.main()
