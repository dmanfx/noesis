from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = ROOT / "DS9" / "gst-plugins"
SOURCE = PLUGIN_ROOT / "noesiseos" / "gstnoesiseos.cpp"
RTSP_RE = re.compile(r"rtsps?://[^\s'\"<>]+", re.IGNORECASE)


def _safe_output(value: str) -> str:
    return RTSP_RE.sub("rtsp://<redacted>", value)


def _load_ds9_pipeline_builder():
    module_name = "ds9_shutdown_graph_characterization"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = ROOT / "DS9" / "noesis" / "pipelines" / "deepstream_pipeline.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load DS9 pipeline builder: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class EosPluginSourceTests(unittest.TestCase):
    def test_source_is_zero_copy_and_uses_standard_downstream_eos(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")

        self.assertIn("gst_event_new_eos()", source)
        self.assertIn(
            "gst_pad_push_event(GST_BASE_TRANSFORM_SRC_PAD(self), event)", source
        )
        self.assertIn('"request-sequence"', source)
        self.assertIn('"accepted-sequence"', source)
        self.assertIn('"last-request-ok"', source)
        self.assertIn("g_param_spec_uint(", source)
        self.assertNotIn("g_param_spec_uint64", source)
        self.assertIn("GST_STATIC_CAPS_ANY", source)
        self.assertIn("gst_base_transform_set_passthrough", source)
        self.assertIn("g_atomic_int_set(&self->eos_accepted, TRUE)", source)
        self.assertIn("g_atomic_int_set(&self->eos_accepted, FALSE)", source)
        self.assertIn("g_atomic_int_get(&self->eos_accepted)", source)
        self.assertIn("GST_BASE_TRANSFORM_FLOW_DROPPED", source)
        self.assertIn("transform_ip_on_passthrough = TRUE", source)
        self.assertIn("g_thread_try_new", source)
        self.assertIn("g_object_ref(G_OBJECT(self))", source)
        self.assertIn("self->request_pending = TRUE", source)
        self.assertNotIn("GST_PAD_STREAM_LOCK", source)
        self.assertLess(
            source.index("g_atomic_int_set(&self->eos_accepted, TRUE)"),
            source.index("g_thread_try_new"),
        )
        self.assertNotIn("gst_buffer_map", source)
        self.assertNotIn("gst_buffer_make_writable", source)

    def test_source_and_build_are_owned_by_ds91(self) -> None:
        self.assertFalse(SOURCE.is_symlink())
        ds9_cmake = (SOURCE.parent / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('NOESIS_DEEPSTREAM_MAJOR="9"', ds9_cmake)
        self.assertNotIn('NOESIS_DEEPSTREAM_MAJOR="8"', ds9_cmake)

    def test_every_declared_runtime_valve_forwards_sticky_eos(self) -> None:
        for relative in (Path("DS9/noesis/pipelines/deepstream_pipeline.py"),):
            with self.subTest(path=relative):
                tree = ast.parse(
                    (ROOT / relative).read_text(encoding="utf-8"),
                    filename=str(relative),
                )
                valve_drop_modes: list[object] = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    if (
                        not isinstance(node.func, ast.Name)
                        or node.func.id != "Component"
                    ):
                        continue
                    keywords = {
                        item.arg: item.value for item in node.keywords if item.arg
                    }
                    element = keywords.get("element")
                    config = keywords.get("config")
                    if (
                        not isinstance(element, ast.Constant)
                        or element.value != "valve"
                    ):
                        continue
                    self.assertIsInstance(
                        config, ast.Dict, f"valve missing config in {relative}"
                    )
                    config_items = {
                        key.value: value
                        for key, value in zip(config.keys, config.values)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    }
                    self.assertIn(
                        "drop-mode",
                        config_items,
                        f"valve missing drop-mode in {relative}",
                    )
                    try:
                        valve_drop_modes.append(
                            ast.literal_eval(config_items["drop-mode"])
                        )
                    except (ValueError, TypeError) as exc:
                        self.fail(f"valve drop-mode is not static in {relative}: {exc}")
                self.assertTrue(valve_drop_modes, f"no valves discovered in {relative}")
                for drop_mode in valve_drop_modes:
                    self.assertIn(
                        drop_mode,
                        (1, 2),
                        f"closed valve can drop sticky EOS in {relative}: {drop_mode}",
                    )


class DS9ShutdownGraphCharacterizationTests(unittest.TestCase):
    def _build_graph(
        self,
        root: Path,
        *,
        preprocess: bool,
        sources: list[dict[str, object]] | None = None,
        loop_local_mp4: bool = True,
    ):
        engine_path = root / "fixture.engine"
        engine_path.write_bytes(b"engine-fixture")
        infer_path = root / "fixture.ini"
        infer_path.write_text(
            "[property]\n"
            "onnx-file=offline-maintenance-source.onnx\n"
            f"model-engine-file={engine_path}\n",
            encoding="utf-8",
        )
        config = {
            "version": 1,
            "batch_size": 1,
            "sources": sources if sources is not None else [
                {
                    "element": "nvurisrcbin",
                    "uri": "file:///tmp/noesis-ds9-graph-fixture.mp4",
                }
            ],
            "streammux": {
                "element": "nvstreammux",
                "batch-size": 1,
                "width": 64,
                "height": 64,
            },
            "preprocess": {"enable": True} if preprocess else {},
            "models": {
                "pgie": {
                    "config-file-path": str(infer_path),
                    "engine": str(engine_path),
                },
                "mapanything": {
                    "enable": True,
                    "config-file-path": str(infer_path),
                    "engine": str(engine_path),
                },
            },
            "tracker": {},
            "analytics": {"enable": False},
            "sinks": [{"name": "test_sink", "type": "fakesink", "sync": False}],
        }
        camera_registry_path = root / "camera-secrets.json"
        camera_registry_path.write_text(
            json.dumps({"version": 1, "sources": {"fixture-camera": "rtsp://example.invalid/live"}}),
            encoding="utf-8",
        )
        camera_registry_path.chmod(0o600)
        config_path = root / "infer.yaml"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        pipeline_module = _load_ds9_pipeline_builder()
        pipeline_module._PIPELINE_SINGLETON = None
        with mock.patch.dict(
            os.environ,
            {
                "NOESIS_DS9_STUB_PIPELINE": "1",
                "NOESIS_BUILD_DIR": str(root / "build"),
                "NOESIS_MOSAIC_RTSP_ENABLED": "0",
                "NOESIS_MOSAIC_WEBRTC_ENABLED": "0",
                "NOESIS_DS9_LOOP_LOCAL_MP4": "1" if loop_local_mp4 else "0",
                "NOESIS_CAMERA_SECRETS_FILE": str(camera_registry_path),
            },
            clear=False,
        ):
            return pipeline_module.build_pipeline(config_path)

    def test_orderly_eos_is_the_first_post_mux_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-eos-graph-") as raw:
            root = Path(raw)
            for preprocess, expected_next in (
                (True, "preprocess"),
                (False, "yolo11_pgie"),
            ):
                with self.subTest(preprocess=preprocess):
                    graph = self._build_graph(root, preprocess=preprocess)
                    eos_name = graph.shutdown_eos_component_name

                    self.assertEqual(eos_name, "orderly_eos_control")
                    self.assertEqual(graph.components[eos_name].element, "noesiseos")
                    self.assertEqual(
                        graph.components["streammux"].downstream, [eos_name]
                    )
                    self.assertEqual(
                        graph.components[eos_name].downstream, [expected_next]
                    )
                    self.assertIn(("streammux", eos_name), graph.ds_pipeline.links)
                    self.assertIn((eos_name, expected_next), graph.ds_pipeline.links)
                    self.assertNotIn(
                        ("streammux", expected_next), graph.ds_pipeline.links
                    )

    def test_only_built_looping_file_sources_declare_hidden_terminal_sinks(self) -> None:
        sources = [
            {"element": "nvurisrcbin", "uri": "file:///tmp/auto-loop.mp4"},
            {"element": "nvurisrcbin", "uri": "file:///tmp/finite.mp4", "file-loop": False},
            {"element": "nvurisrcbin", "uri": "file:///tmp/explicit-loop.mkv", "file-loop": True},
            {"element": "nvurisrcbin", "uri_secret": "fixture-camera", "file-loop": True},
            {"element": "uridecodebin", "uri": "file:///tmp/other-element.mp4"},
        ]
        with tempfile.TemporaryDirectory(prefix="ds9-eos-source-sinks-") as raw:
            for loop_local_mp4, expected in (
                (True, ["source_0/nvurisrc_bin__fakesink", "source_2/nvurisrc_bin__fakesink"]),
                (False, ["source_2/nvurisrc_bin__fakesink"]),
            ):
                with self.subTest(loop_local_mp4=loop_local_mp4):
                    graph = self._build_graph(
                        Path(raw), preprocess=False, sources=sources,
                        loop_local_mp4=loop_local_mp4,
                    )
                    bridge = graph.components[graph.shutdown_eos_component_name]
                    self.assertEqual(
                        bridge.config.get("upstream-sink-paths", ""), ",".join(expected)
                    )
                    for path in expected:
                        source = graph.components[path.split("/")[0]]
                        self.assertEqual(source.element, "nvurisrcbin")
                        self.assertTrue(source.config["file-loop"])
                        self.assertTrue(str(source.config["uri"]).startswith("file:"))

    def test_live_and_finite_file_graphs_keep_empty_upstream_sink_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-eos-no-hidden-sinks-") as raw:
            graph = self._build_graph(
                Path(raw), preprocess=False,
                sources=[
                    {"element": "nvurisrcbin", "uri_secret": "fixture-camera"},
                    {"element": "nvurisrcbin", "uri": "file:///tmp/finite.mp4", "file-loop": False},
                ],
            )
        bridge = graph.components[graph.shutdown_eos_component_name]
        self.assertEqual(bridge.config.get("upstream-sink-paths", ""), "")

    def test_every_potentially_closed_ds9_valve_forwards_sticky_eos(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ds9-eos-valves-") as raw:
            graph = self._build_graph(Path(raw), preprocess=True)
        valves = [
            component
            for component in graph.components.values()
            if component.element == "valve"
        ]
        self.assertTrue(valves)
        for valve in valves:
            self.assertIn(
                int(valve.config.get("drop-mode", 0)),
                (1, 2),
                f"{valve.name} can suppress EOS while closed",
            )


class EosPluginRuntimeTests(unittest.TestCase):
    def _plugin_binary(self, plugin_root: Path) -> Path:
        binary = plugin_root / "libgstnoesiseos.so"
        if not binary.is_file():
            self.skipTest(f"plugin has not been built: {binary}")
        return binary

    def _env(self, plugin_root: Path, registry_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["GST_PLUGIN_PATH"] = str(plugin_root)
        env["GST_REGISTRY"] = str(registry_dir / "registry.bin")
        return env

    def _run_bounded(
        self,
        program: str,
        *,
        timeout: float,
        plugin_root: Path = PLUGIN_ROOT,
    ) -> tuple[int, str, bool]:
        self._plugin_binary(plugin_root)
        with tempfile.TemporaryDirectory(prefix="noesis-eos-process-") as raw:
            process = subprocess.Popen(
                [sys.executable, "-c", program],
                cwd=ROOT,
                env=self._env(plugin_root, Path(raw)),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                output, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    output, _ = process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    output, _ = process.communicate(timeout=2.0)

            deadline = time.monotonic() + 2.0
            while True:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                if time.monotonic() >= deadline:
                    self.fail(
                        f"Service Maker process group leaked after exit: {process.pid}"
                    )
                time.sleep(0.05)

        return int(process.returncode or 0), output, timed_out

    @staticmethod
    def _json_lines(output: str) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in output.splitlines()
            if line.startswith("{") and line.endswith("}")
        ]

    def test_binary_registers_owned_uint_contracts(self) -> None:
        binary = self._plugin_binary(PLUGIN_ROOT)
        with tempfile.TemporaryDirectory(prefix="noesis-eos-inspect-") as raw:
            result = subprocess.run(
                ["gst-inspect-1.0", "noesiseos"],
                cwd=ROOT,
                env=self._env(PLUGIN_ROOT, Path(raw)),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(
            result.returncode,
            0,
            _safe_output(result.stdout + result.stderr),
        )
        self.assertIn(str(binary), result.stdout)
        self.assertIn("orderly EOS control bridge (DS9)", result.stdout)
        self.assertIn("request-sequence", result.stdout)
        self.assertIn("accepted-sequence", result.stdout)
        self.assertIn("last-request-ok", result.stdout)
        self.assertIn("Unsigned Integer. Range: 0 - 4294967295", result.stdout)

    def test_monotonic_request_emits_exactly_one_eos_and_acknowledges_it(self) -> None:
        program = textwrap.dedent(
            """
            import json
            import threading
            import time
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            pipeline = Gst.parse_launch(
                "appsrc name=source is-live=true format=time "
                "caps=video/x-raw,format=I420,width=16,height=16,framerate=1/1 "
                "! noesiseos name=quiesce ! fakesink name=sink sync=false"
            )
            quiesce = pipeline.get_by_name("quiesce")
            sink = pipeline.get_by_name("sink")
            eos_seen = []
            upstream_buffers = []
            downstream_buffers = []
            eos_entered = threading.Event()
            eos_release = threading.Event()

            def event_probe(_pad, info):
                event = info.get_event()
                if event is not None and event.type == Gst.EventType.EOS:
                    eos_seen.append("eos")
                    eos_entered.set()
                    assert eos_release.wait(3.0)
                return Gst.PadProbeReturn.OK

            def upstream_probe(_pad, info):
                buffer = info.get_buffer()
                if buffer is not None:
                    upstream_buffers.append(buffer.pts)
                return Gst.PadProbeReturn.OK

            def downstream_probe(_pad, info):
                buffer = info.get_buffer()
                if buffer is not None:
                    downstream_buffers.append(buffer.pts)
                return Gst.PadProbeReturn.OK

            quiesce.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, upstream_probe)
            sink.get_static_pad("sink").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM, event_probe)
            sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, downstream_probe)
            assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
            source = pipeline.get_by_name("source")
            buffer = Gst.Buffer.new_allocate(None, 16 * 16 * 3 // 2, None)
            buffer.pts = 0
            buffer.duration = Gst.SECOND
            assert source.emit("push-buffer", buffer) == Gst.FlowReturn.OK
            pipeline.get_state(2 * Gst.SECOND)
            setter_started = time.monotonic()
            quiesce.set_property("request-sequence", 7)
            setter_elapsed = time.monotonic() - setter_started
            assert eos_entered.wait(1.0)
            pending = [
                quiesce.get_property("request-sequence"),
                quiesce.get_property("accepted-sequence"),
                quiesce.get_property("last-request-ok"),
            ]
            quiesce.set_property("request-sequence", 7)
            quiesce.set_property("request-sequence", 6)
            late_pending = Gst.Buffer.new_allocate(None, 16 * 16 * 3 // 2, None)
            late_pending.pts = Gst.SECOND
            late_pending.duration = Gst.SECOND
            late_pending_flow = quiesce.get_static_pad("sink").chain(late_pending)
            eos_release.set()
            message = pipeline.get_bus().timed_pop_filtered(
                2 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            deadline = time.monotonic() + 2.0
            while quiesce.get_property("accepted-sequence") != 7 and time.monotonic() < deadline:
                time.sleep(0.01)
            late_accepted = Gst.Buffer.new_allocate(None, 16 * 16 * 3 // 2, None)
            late_accepted.pts = 2 * Gst.SECOND
            late_accepted.duration = Gst.SECOND
            late_accepted_flow = quiesce.get_static_pad("sink").chain(late_accepted)
            payload = {
                "request": quiesce.get_property("request-sequence"),
                "accepted": quiesce.get_property("accepted-sequence"),
                "ok": quiesce.get_property("last-request-ok"),
                "events": eos_seen,
                "bus_eos": message is not None and message.type == Gst.MessageType.EOS,
                "pending": pending,
                "setter_elapsed_s": setter_elapsed,
                "late_pending_flow": int(late_pending_flow),
                "late_accepted_flow": int(late_accepted_flow),
                "upstream_buffers": upstream_buffers,
                "downstream_buffers": downstream_buffers,
            }
            pipeline.set_state(Gst.State.NULL)
            print(json.dumps(payload, sort_keys=True))
            """
        )
        returncode, output, timed_out = self._run_bounded(program, timeout=10.0)
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payload = self._json_lines(output)[-1]
        self.assertEqual(payload["accepted"], 7)
        self.assertTrue(payload["bus_eos"])
        self.assertEqual(payload["events"], ["eos"])
        self.assertEqual(payload["pending"], [7, 0, False])
        self.assertLess(float(payload["setter_elapsed_s"]), 0.2)
        self.assertEqual(payload["late_pending_flow"], 0)
        self.assertEqual(payload["late_accepted_flow"], 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["request"], 7)
        self.assertEqual(payload["upstream_buffers"], [0, 1_000_000_000, 2_000_000_000])
        self.assertEqual(payload["downstream_buffers"], [0])

    @staticmethod
    def _hidden_sink_program(mode: str, target_paths: str = "") -> str:
        """Model nvurisrcbin's terminal side branch before the post-mux bridge."""
        return textwrap.dedent(
            """
            import json
            import threading
            import time
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            MODE = __MODE__
            TARGET_PATHS = __TARGET_PATHS__
            pipeline = Gst.Pipeline.new("nested-source-eos")
            source_bin = Gst.Bin.new("source_0")
            source = Gst.ElementFactory.make("appsrc", "input")
            split = Gst.ElementFactory.make("tee", "split")
            main_output = Gst.ElementFactory.make("identity", "main_output")
            hidden = Gst.ElementFactory.make("fakesink", "nvurisrc_bin__fakesink")
            bridge = Gst.ElementFactory.make("noesiseos", "quiesce")
            main = Gst.ElementFactory.make("fakesink", "main_sink")
            source.set_property("is-live", True)
            source.set_property("format", Gst.Format.TIME)
            source.set_property("caps", Gst.Caps.from_string("application/octet-stream"))
            for sink in (hidden, main):
                sink.set_property("sync", False)
                sink.set_property("async", False)
                sink.set_property("signal-handoffs", True)
            for element in (source, split, main_output, hidden):
                source_bin.add(element)
            if TARGET_PATHS == "source_0/disconnected_sink":
                disconnected = Gst.ElementFactory.make("fakesink", "disconnected_sink")
                disconnected.set_property("async", False)
                source_bin.add(disconnected)
            assert source.link(split)
            assert split.link(hidden)
            assert split.link(main_output)
            source_bin.add_pad(Gst.GhostPad.new("src", main_output.get_static_pad("src")))
            for element in (source_bin, bridge, main):
                pipeline.add(element)
            assert source_bin.link(bridge)
            assert bridge.link(main)
            if TARGET_PATHS:
                bridge.set_property("upstream-sink-paths", TARGET_PATHS)

            rendered = {"main": [], "hidden": []}
            events = {"main": [], "hidden": []}
            main_eos_entered = threading.Event()
            release_main_eos = threading.Event()
            def handoff(_sink, buffer, _pad, name):
                rendered[name].append(int(buffer.pts))
            def event_probe(_pad, info, name):
                event = info.get_event()
                if event is not None and event.type == Gst.EventType.EOS:
                    events[name].append("eos")
                    if name == "main" and MODE == "targeted":
                        main_eos_entered.set()
                        assert release_main_eos.wait(3.0)
                return Gst.PadProbeReturn.OK
            for name, sink in (("main", main), ("hidden", hidden)):
                sink.connect("handoff", handoff, name)
                sink.get_static_pad("sink").add_probe(
                    Gst.PadProbeType.EVENT_DOWNSTREAM, event_probe, name
                )
            def buffer(pts):
                item = Gst.Buffer.new_allocate(None, 16, None)
                item.pts = pts
                item.duration = Gst.SECOND
                return item
            def await_render(pts):
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if all(pts in values for values in rendered.values()):
                        return
                    time.sleep(0.005)
                raise AssertionError(rendered)
            assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
            assert source.emit("push-buffer", buffer(0)) == Gst.FlowReturn.OK
            await_render(0)
            pipeline.get_state(Gst.SECOND)
            started = time.monotonic()
            bridge.set_property("request-sequence", 37)
            setter_elapsed = time.monotonic() - started
            pending = None
            late_flows = []
            if MODE == "targeted":
                assert main_eos_entered.wait(1.0)
                pending = [bridge.get_property("accepted-sequence"), bridge.get_property("last-request-ok")]
                # The hidden sink's terminal guard must already be active while
                # the main EOS is draining, before hidden EOS is delivered.
                late_flows.append(int(hidden.get_static_pad("sink").get_peer().push(buffer(Gst.SECOND))))
                late_flows.append(int(bridge.get_static_pad("sink").chain(buffer(Gst.SECOND))))
                release_main_eos.set()
            if MODE == "invalid":
                # A bad path must roll back before either branch gets EOS.
                # Rendering proves the rejected request also removed the drop barrier.
                time.sleep(0.10)
                assert source.emit("push-buffer", buffer(Gst.SECOND)) == Gst.FlowReturn.OK
                await_render(Gst.SECOND)
            else:
                deadline = time.monotonic() + 2.0
                while bridge.get_property("accepted-sequence") != 37 and time.monotonic() < deadline:
                    time.sleep(0.005)

            wait_result = []
            def wait_for_pipeline_eos():
                message = pipeline.get_bus().timed_pop_filtered(
                    (2 * Gst.SECOND if MODE == "targeted" else Gst.SECOND // 4),
                    Gst.MessageType.EOS | Gst.MessageType.ERROR,
                )
                wait_result.append(
                    "none" if message is None else "eos" if message.type == Gst.MessageType.EOS else "error"
                )
            waiter = threading.Thread(target=wait_for_pipeline_eos)
            waiter.start()
            waiter.join(3.0)
            assert not waiter.is_alive()
            if MODE == "targeted":
                # Push from the actual terminal branch peer, just as its tee
                # would. Late buffers/lists must not poison upstream with EOS.
                late_flows.append(int(hidden.get_static_pad("sink").get_peer().push(buffer(2 * Gst.SECOND))))
                batch = Gst.BufferList.new()
                batch.insert(-1, buffer(3 * Gst.SECOND))
                late_flows.append(int(hidden.get_static_pad("sink").get_peer().push_list(batch)))
                late_flows.append(int(bridge.get_static_pad("sink").chain(buffer(2 * Gst.SECOND))))
            payload = {
                "request": bridge.get_property("request-sequence"),
                "accepted": bridge.get_property("accepted-sequence"),
                "ok": bridge.get_property("last-request-ok"),
                "setter_elapsed_s": setter_elapsed,
                "pending": pending,
                "events": events,
                "rendered": rendered,
                "late_flows": late_flows,
                "wait_result": wait_result,
                "wait_completed_before_null": not waiter.is_alive(),
            }
            pipeline.set_state(Gst.State.NULL)
            print(json.dumps(payload, sort_keys=True))
            """
        ).replace("__MODE__", repr(mode)).replace("__TARGET_PATHS__", repr(target_paths))

    def test_hidden_source_sink_blocks_aggregate_eos_with_default_bridge(self) -> None:
        returncode, output, timed_out = self._run_bounded(
            self._hidden_sink_program("default"), timeout=8.0
        )
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payload = self._json_lines(output)[-1]
        self.assertEqual(payload["accepted"], 37)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["events"], {"main": ["eos"], "hidden": []})
        self.assertEqual(payload["wait_result"], ["none"])
        self.assertEqual(payload["rendered"], {"main": [0], "hidden": [0]})

    def test_declared_hidden_sink_completes_aggregate_eos_and_absorbs_late_buffers(self) -> None:
        returncode, output, timed_out = self._run_bounded(
            self._hidden_sink_program("targeted", "source_0/nvurisrc_bin__fakesink"),
            timeout=8.0,
        )
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payload = self._json_lines(output)[-1]
        self.assertEqual(payload["request"], 37)
        self.assertEqual(payload["accepted"], 37)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["pending"], [0, False])
        self.assertEqual(payload["events"], {"main": ["eos"], "hidden": ["eos"]})
        self.assertEqual(payload["wait_result"], ["eos"])
        self.assertTrue(payload["wait_completed_before_null"])
        self.assertEqual(payload["rendered"], {"main": [0], "hidden": [0]})
        self.assertEqual(payload["late_flows"], [0, 0, 0, 0, 0])
        self.assertLess(float(payload["setter_elapsed_s"]), 0.2)

    def test_invalid_hidden_sink_paths_reject_before_eos_and_preserve_flow(self) -> None:
        for path in (
            "source_0/missing",
            "source_0/main_output",
            "source_0/nvurisrc_bin__fakesink,source_0/missing",
            "nvurisrc_bin__fakesink",  # Skipping source_0 must not resolve recursively.
            "source_0/nvurisrc_bin__fakesink,source_0/nvurisrc_bin__fakesink",
            "source_0/disconnected_sink",
        ):
            with self.subTest(path=path):
                returncode, output, timed_out = self._run_bounded(
                    self._hidden_sink_program("invalid", path), timeout=8.0
                )
                self.assertFalse(timed_out, _safe_output(output))
                self.assertEqual(returncode, 0, _safe_output(output))
                payload = self._json_lines(output)[-1]
                self.assertEqual(payload["request"], 37)
                self.assertEqual(payload["accepted"], 0)
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["events"], {"main": [], "hidden": []})
                self.assertEqual(payload["wait_result"], ["none"])
                self.assertEqual(
                    payload["rendered"], {"main": [0, 1_000_000_000], "hidden": [0, 1_000_000_000]}
                )
                self.assertLess(float(payload["setter_elapsed_s"]), 0.2)

    def test_offline_eos_failure_rolls_back_terminal_drop(self) -> None:
        program = textwrap.dedent(
            """
            import json
            import time
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            pipeline = Gst.Pipeline.new("offline-eos")
            source = Gst.ElementFactory.make("appsrc", "source")
            quiesce = Gst.ElementFactory.make("noesiseos", "quiesce")
            sink = Gst.ElementFactory.make("fakesink", "sink")
            source.set_property("is-live", True)
            source.set_property("format", Gst.Format.TIME)
            sink.set_property("sync", False)
            sink.set_property("async", False)
            for element in (source, quiesce, sink):
                pipeline.add(element)
            assert source.link(quiesce)
            seen = []

            def probe(_pad, info):
                buffer = info.get_buffer()
                if buffer is not None:
                    seen.append(buffer.pts)
                return Gst.PadProbeReturn.OK

            sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, probe)
            started = time.monotonic()
            quiesce.set_property("request-sequence", 1)
            setter_elapsed = time.monotonic() - started
            time.sleep(0.2)
            assert quiesce.link(sink)
            assert pipeline.set_state(Gst.State.PLAYING) != Gst.StateChangeReturn.FAILURE
            buffer = Gst.Buffer.new_allocate(None, 16, None)
            buffer.pts = 123
            flow = quiesce.get_static_pad("sink").chain(buffer)
            time.sleep(0.05)
            payload = {
                "request": quiesce.get_property("request-sequence"),
                "accepted": quiesce.get_property("accepted-sequence"),
                "ok": quiesce.get_property("last-request-ok"),
                "setter_elapsed_s": setter_elapsed,
                "flow": int(flow),
                "downstream_buffers": seen,
            }
            pipeline.set_state(Gst.State.NULL)
            print(json.dumps(payload, sort_keys=True))
            """
        )
        returncode, output, timed_out = self._run_bounded(program, timeout=10.0)
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payload = self._json_lines(output)[-1]
        self.assertEqual(payload["request"], 1)
        self.assertEqual(payload["accepted"], 0)
        self.assertFalse(payload["ok"])
        self.assertLess(float(payload["setter_elapsed_s"]), 0.2)
        self.assertEqual(payload["flow"], 0)
        self.assertEqual(payload["downstream_buffers"], [123])

    def test_owner_rtsp_reconnect_graph_waits_without_pipeline_stop(self) -> None:
        if not all(
            path.is_char_device()
            for path in (Path("/dev/nvidia0"), Path("/dev/nvidiactl"))
        ):
            self.skipTest(
                "owner RTSP reconnect characterization requires an NVIDIA device"
            )
        program = textwrap.dedent(
            """
            import json
            import time
            from pyservicemaker import Pipeline
            from noesis_core.runtime_secrets import RuntimeSecretError, load_pipeline_config

            try:
                uri = load_pipeline_config("config/infer.yaml")["sources"][0]["uri"]
            except RuntimeSecretError:
                raise SystemExit(77)
            pipeline = Pipeline("noesis-eos-owner-rtsp")
            pipeline.add("nvurisrcbin", "source", {
                "uri": uri,
                "rtsp-reconnect-interval": 10,
                "init-rtsp-reconnect-interval": 5,
                "rtsp-reconnect-attempts": 4,
                "latency": 100,
                "select-rtp-protocol": 4,
                "disable-audio": True,
                "cudadec-memtype": 0,
                "gpu-id": 0,
            })
            pipeline.add("noesiseos", "quiesce")
            pipeline.add("fakesink", "sink", {"sync": False})
            pipeline.link("source", "quiesce", "sink")
            pipeline.prepare()
            source = pipeline["source"]
            quiesce = pipeline["quiesce"]
            assert source.get("rtsp-reconnect-interval") == 10
            assert source.get("init-rtsp-reconnect-interval") == 5
            assert source.get("rtsp-reconnect-attempts") == 4
            pipeline.activate()
            time.sleep(2.0)
            started = time.monotonic()
            quiesce.set({"request-sequence": 17})
            setter_elapsed = time.monotonic() - started
            deadline = time.monotonic() + 5.0
            while quiesce.get("accepted-sequence") != 17 and time.monotonic() < deadline:
                time.sleep(0.01)
            accepted = [
                quiesce.get("request-sequence"),
                quiesce.get("accepted-sequence"),
                quiesce.get("last-request-ok"),
            ]
            pipeline.wait()
            print(json.dumps({
                "accepted": accepted,
                "wait_elapsed_s": time.monotonic() - started,
                "setter_elapsed_s": setter_elapsed,
                "returned_without_stop": True,
            }, sort_keys=True), flush=True)
            """
        )
        self.assertNotIn("pipeline.stop", program.lower())
        returncode, output, timed_out = self._run_bounded(program, timeout=15.0)
        if returncode == 77:
            self.skipTest("owner camera source secrets are unavailable")
        self.assertNotIn("rtsp://", output.lower(), "RTSP URI leaked into smoke output")
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payload = self._json_lines(output)[-1]
        self.assertEqual(payload["accepted"], [17, 17, True])
        self.assertTrue(payload["returned_without_stop"])
        self.assertLess(float(payload["setter_elapsed_s"]), 0.2)
        self.assertLess(float(payload["wait_elapsed_s"]), 5.0)

    @staticmethod
    def _valve_program(mode: str) -> str:
        return textwrap.dedent(
            f"""
            import json
            import time
            from pyservicemaker import Pipeline

            mode = {mode!r}
            drop_mode = 1 if mode == "forward-sticky" else 0
            pipeline = Pipeline("noesis-eos-valve-" + mode)
            for element, name, properties in (
                ("videotestsrc", "source", {{"is-live": True}}),
                ("noesiseos", "quiesce", {{}}),
                ("tee", "split", {{}}),
                ("queue", "open_queue", {{}}),
                ("fakesink", "open_sink", {{"sync": False}}),
                ("queue", "gated_queue", {{}}),
                ("valve", "gate", {{"drop": False, "drop-mode": drop_mode}}),
                ("fakesink", "gated_sink", {{"sync": False}}),
            ):
                pipeline.add(element, name, properties)
            pipeline.link("source", "quiesce", "split")
            pipeline.link("split", "open_queue", "open_sink")
            pipeline.link("split", "gated_queue", "gate", "gated_sink")
            pipeline.prepare()
            pipeline.activate()
            time.sleep(0.5)
            gate = pipeline["gate"]
            gate.set({{"drop": True}})
            assert gate.get("drop") is True
            if mode == "drop-all-open":
                gate.set({{"drop": False}})
                assert gate.get("drop") is False
            quiesce = pipeline["quiesce"]
            started = time.monotonic()
            quiesce.set({{"request-sequence": 29}})
            setter_elapsed = time.monotonic() - started
            deadline = time.monotonic() + 5.0
            while quiesce.get("accepted-sequence") != 29 and time.monotonic() < deadline:
                time.sleep(0.01)
            accepted = [
                quiesce.get("request-sequence"),
                quiesce.get("accepted-sequence"),
                quiesce.get("last-request-ok"),
            ]
            print(json.dumps({{
                "phase": "accepted",
                "accepted": accepted,
                "setter_elapsed_s": setter_elapsed,
            }}, sort_keys=True), flush=True)
            pipeline.wait()
            print(json.dumps({{
                "phase": "wait-returned",
                "wait_elapsed_s": time.monotonic() - started,
            }}, sort_keys=True), flush=True)
            """
        )

    def test_forward_sticky_closed_valve_preserves_eos(self) -> None:
        program = self._valve_program("forward-sticky")
        self.assertNotIn("pipeline.stop", program.lower())
        returncode, output, timed_out = self._run_bounded(program, timeout=10.0)
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payloads = self._json_lines(output)
        self.assertEqual(payloads[0]["accepted"], [29, 29, True])
        self.assertLess(float(payloads[0]["setter_elapsed_s"]), 0.2)
        self.assertEqual(payloads[-1]["phase"], "wait-returned")
        self.assertLess(float(payloads[-1]["wait_elapsed_s"]), 5.0)

    def test_drop_all_closed_valve_blocks_global_eos_despite_local_ack(self) -> None:
        program = self._valve_program("drop-all-closed")
        returncode, output, timed_out = self._run_bounded(program, timeout=3.0)
        self.assertTrue(timed_out, _safe_output(output))
        self.assertNotEqual(returncode, 0)
        payloads = self._json_lines(output)
        self.assertEqual(payloads[-1]["accepted"], [29, 29, True])
        self.assertEqual(payloads[-1]["phase"], "accepted")
        self.assertLess(float(payloads[-1]["setter_elapsed_s"]), 0.2)

    def test_opening_drop_all_valve_before_request_restores_global_eos(self) -> None:
        program = self._valve_program("drop-all-open")
        self.assertNotIn("pipeline.stop", program.lower())
        returncode, output, timed_out = self._run_bounded(program, timeout=10.0)
        self.assertFalse(timed_out, _safe_output(output))
        self.assertEqual(returncode, 0, _safe_output(output))
        payloads = self._json_lines(output)
        self.assertEqual(payloads[0]["accepted"], [29, 29, True])
        self.assertLess(float(payloads[0]["setter_elapsed_s"]), 0.2)
        self.assertEqual(payloads[-1]["phase"], "wait-returned")
        self.assertLess(float(payloads[-1]["wait_elapsed_s"]), 5.0)


if __name__ == "__main__":
    unittest.main()
