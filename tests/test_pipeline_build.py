from __future__ import annotations

import configparser
import importlib.util
import sys
from pathlib import Path
from typing import Callable, List

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noesis.pipelines import ds8_pipeline as pipeline  # noqa: E402
from noesis_core.inference_runtime_contract import (  # noqa: E402
    materialize_nvinfer_engine_only_config,
)


@pytest.fixture(autouse=True)
def reset_pipeline_singleton(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Ensure each test starts with a fresh pipeline graph."""
    monkeypatch.setenv("NOESIS_MOSAIC_H264_SHM", str(tmp_path / "mosaic-h264.sock"))
    pipeline._PIPELINE_SINGLETON = None
    yield
    pipeline._PIPELINE_SINGLETON = None


def _infer_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "infer.yaml"


def test_active_pipeline_sources_are_checkout_portable():
    active_yamls = [
        ROOT / "config" / "infer.yaml",
        ROOT / "config" / "infer_v3dt_baseline.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_fast1056_mp4.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_fast_inv1_mp4.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_fast_mp4.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_mp4.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_noreid_mp4.yaml",
        ROOT / "config" / "infer_v3dt_reimpl_posefast_mp4.yaml",
        ROOT / "config" / "infer_reid_rtsp_min.yaml",
        ROOT / "config" / "infer_smoke_reid.yaml",
        ROOT / "DS9" / "config" / "infer.yaml",
        ROOT / "DS9" / "config" / "infer_v3dt.yaml",
    ]
    expected_depth_sources = {
        "pipelines/config_infer_secondary_depth_tracking_da2.ini",
        "DS9/pipelines/config_infer_secondary_depth_tracking_da2.ini",
    }

    for path in active_yamls:
        text = path.read_text(encoding="utf-8")
        assert "/home/" not in text, path
        payload = yaml.safe_load(text)
        models = payload.get("models") or {}
        depth = models.get("depth_tracking") or {}
        if depth:
            source = str(depth.get("config-file-path") or "")
            assert source in expected_depth_sources, path
            assert (ROOT / source).is_file(), path


def test_active_mosaic_configs_share_shm_quality_and_recovery_contract() -> None:
    active_yamls = (
        ROOT / "config" / "infer.yaml",
        ROOT / "config" / "infer_v3dt_baseline.yaml",
        *sorted((ROOT / "config").glob("infer_v3dt_reimpl*.yaml")),
        ROOT / "config" / "infer_reid_rtsp_min.yaml",
        ROOT / "config" / "infer_smoke_reid.yaml",
        ROOT / "DS9" / "config" / "infer.yaml",
        ROOT / "DS9" / "config" / "infer_v3dt.yaml",
    )

    for path in active_yamls:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        mosaic = payload["mosaic_output"]
        assert mosaic["rtsp_enabled"] is False, path
        assert mosaic["mosaic_h264_shm_socket"] == "/tmp/noesis-mosaic-h264", path
        assert mosaic["encoder"] == "nvv4l2h264enc", path
        assert int(mosaic["video_bitrate_kbps"]) == 12_000, path
        assert int(mosaic["h264_iframeinterval"]) == 10, path
        assert int(mosaic["h264_idrinterval"]) == 10, path
        if path.name in {"infer_reid_rtsp_min.yaml", "infer_smoke_reid.yaml"}:
            assert mosaic["mosaic_webrtc_enabled"] is False, path
        else:
            assert mosaic["mosaic_webrtc_enabled"] is True, path
        assert not {
            "rtsp_iframeinterval",
            "rtsp_idrinterval",
            "rtsp_profile",
            "rtsp_preset_id",
        }.intersection(mosaic), path


def test_retired_rtsp_prefixed_h264_keys_fail_closed(tmp_path: Path) -> None:
    payload = yaml.safe_load(_infer_config_path().read_text(encoding="utf-8"))
    payload["mosaic_output"]["rtsp_idrinterval"] = 1
    config_path = tmp_path / "retired-h264-key.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="retired RTSP-prefixed H.264 keys"):
        pipeline.build_pipeline(config_path)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    (
        ("video_bitrate_kbps", 0, "video_bitrate_kbps must be between"),
        ("encoder", "", "must be nvv4l2h264enc"),
        ("h264_iframeinterval", 0, "iframeinterval and idrinterval must be positive"),
        ("h264_idrinterval", 0, "iframeinterval and idrinterval must be positive"),
        ("h264_preset_id", 0, "h264_preset_id must be between"),
        ("mosaic_h264_shm_size_bytes", 0, "must be between 4 MiB"),
    ),
)
def test_explicit_invalid_mosaic_h264_values_fail_closed(
    tmp_path: Path,
    key: str,
    value: object,
    error: str,
) -> None:
    payload = yaml.safe_load(_infer_config_path().read_text(encoding="utf-8"))
    payload["mosaic_output"][key] = value
    config_path = tmp_path / f"invalid-{key}.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        pipeline.build_pipeline(config_path)


def test_reviewed_depth_source_derives_engine_only_config(tmp_path: Path):
    source = ROOT / "pipelines" / "config_infer_secondary_depth_tracking_da2.ini"
    engine = (
        ROOT
        / "models"
        / "engines"
        / "depth_anything_v2_metric_hypersim_vits_294x518_b3_fp16.engine"
    )
    derived = materialize_nvinfer_engine_only_config(
        source_config=source,
        engine_path=engine,
        output_root=tmp_path,
        component_name="depth_tracking_fullframe",
        repo_root=ROOT,
    )
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(derived, encoding="utf-8")
    props = parser["property"]

    assert "onnx-file" not in props
    assert Path(props["model-engine-file"]) == engine.resolve()
    assert int(props["batch-size"]) == 3
    assert int(props["interval"]) == 1
    assert int(props["gie-unique-id"]) == 5


def test_canonical_build_uses_reviewed_sources_with_empty_external_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    build_root = tmp_path / "runtime-build"
    monkeypatch.setenv("NOESIS_BUILD_DIR", str(build_root))

    graph = pipeline.build_pipeline(_infer_config_path())

    for component_name in (
        "yolo11_pgie",
        "reid_sgie",
        "yolo26_pose",
        "depth_tracking_fullframe",
        "mapanything_fullframe",
    ):
        config_path = Path(graph.components[component_name].config["config-file-path"])
        assert config_path.is_relative_to(build_root / "runtime_inference")
        assert config_path.is_file()

    pose_parser = configparser.ConfigParser(interpolation=None, strict=False)
    pose_parser.read(
        graph.components["yolo26_pose"].config["config-file-path"],
        encoding="utf-8",
    )
    pose_props = pose_parser["property"]
    assert "onnx-file" not in pose_props
    assert Path(pose_props["labelfile-path"]) == (
        ROOT / "testpipelines" / "yolo26-pose" / "labels.txt"
    ).resolve()
    assert Path(pose_props["model-engine-file"]) == (
        ROOT / "models" / "engines" / "yolo26n-pose_b3_fp16.engine"
    ).resolve()

    map_parser = configparser.ConfigParser(interpolation=None, strict=False)
    map_parser.read(
        graph.components["mapanything_fullframe"].config["config-file-path"],
        encoding="utf-8",
    )
    map_props = map_parser["property"]
    assert Path(map_props["model-engine-file"]) == (
        ROOT / "models" / "mapanything_depth" / "1" / "model.plan"
    ).resolve()
    assert not (build_root / "config_infer_depth_tracking_da2_vits_294x518_b3_i1.ini").exists()


def test_runtime_buffer_operators_fail_loudly_without_servicemaker(monkeypatch):
    graph = pipeline.DS8Pipeline(yaml_path=_infer_config_path(), config={})
    monkeypatch.setattr(pipeline, "BufferOperator", None)

    with pytest.raises(RuntimeError, match="BufferOperator is unavailable"):
        pipeline.DepthGateOperator(graph)
    with pytest.raises(RuntimeError, match="BufferOperator is unavailable"):
        pipeline.LatencyProbeOperator(graph)


def test_enabled_dewarpers_use_full_output_surface():
    config_path = _infer_config_path()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for source in cfg["sources"]:
        dewarper = source.get("dewarper") or {}
        if not dewarper.get("enable"):
            continue
        dewarper_path = ROOT / dewarper["config-file"]
        parser = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False)
        parser.optionxform = str
        parser.read(dewarper_path, encoding="utf-8")
        props = parser["property"]
        surface = parser["surface0"]
        assert int(float(surface["width"])) == int(float(props["output-width"]))
        assert int(float(surface["height"])) == int(float(props["output-height"]))


def test_build_pipeline_graph_from_yaml():
    config_path = _infer_config_path()
    graph = pipeline.build_pipeline(config_path)

    assert isinstance(graph, pipeline.DS8Pipeline)
    assert pipeline.get_pipeline() is graph
    assert graph.depth_enabled is False
    assert graph.valve_name == "mapanything_valve"

    required = {
        "streammux",
        "yolo11_pgie",
        "orderly_eos_control",
        "main_tee",
        "tracker",
        "reid_sgie",
        "analytics",
        "depth_tracking_queue",
        "depth_tracking_fullframe",
        "mapanything_queue",
        "mapanything_valve",
        "mapanything_fullframe",
        "world_observation_stage",
        "tracking_telemetry_stage",
        "osd",
        "mosaic_encode_queue",
        "mosaic_encoder_caps",
        "mosaic_force_idr",
        "mosaic_h264_encoder",
        "mosaic_h264_parse",
        "mosaic_h264_au_caps",
        "mosaic_webrtc_au_queue",
        "mosaic_h264_shmsink",
    }
    assert required.issubset(graph.components.keys())

    valve = graph.components["mapanything_valve"]
    assert valve.element == "valve"
    assert valve.config["drop"] is True
    tee = graph.components["main_tee"]
    orderly_eos = graph.components["orderly_eos_control"]
    assert orderly_eos.element == "noesiseos"
    assert graph.components["streammux"].downstream == ["orderly_eos_control"]
    assert orderly_eos.downstream == ["preprocess"]
    assert graph.components["yolo11_pgie"].downstream == ["main_tee"]
    assert tee.downstream == ["analytics_exclude", "depth_tracking_queue", "mapanything_queue"]
    assert graph.components["analytics_exclude"].downstream == ["tracker"]
    assert graph.components["tracker"].downstream == ["analytics"]
    assert graph.components["analytics"].downstream == ["reid_sgie"]
    assert graph.components["reid_sgie"].downstream == ["yolo26_pose"]
    assert graph.components["yolo26_pose"].downstream == ["world_observation_stage"]
    assert graph.components["world_observation_stage"].downstream == ["tracking_telemetry_stage"]
    assert graph.components["tracking_telemetry_stage"].downstream == ["tiler"]
    assert graph.ds_pipeline is not None
    assert ((("dewarper_caps_out_0", "streammux"), ("", "sink_%u"))) in graph.ds_pipeline.links
    assert ((("dewarper_caps_out_1", "streammux"), ("", "sink_%u"))) in graph.ds_pipeline.links
    assert ((("dewarper_caps_out_2", "streammux"), ("", "sink_%u"))) in graph.ds_pipeline.links
    depth_tracking = graph.components["depth_tracking_fullframe"]
    assert graph.components["depth_tracking_queue"].downstream == ["depth_tracking_fullframe"]
    assert depth_tracking.downstream == ["depth_tracking_fullframe_sink"]
    ma = graph.components["mapanything_fullframe"]
    assert graph.components["mapanything_queue"].downstream == ["mapanything_valve"]
    assert graph.components["mapanything_valve"].downstream == ["mapanything_fullframe"]
    assert ma.downstream == ["mapanything_fullframe_sink"]
    assert (("sink_tee", "mosaic_encode_queue")) in graph.ds_pipeline.links
    assert (("mosaic_encode_vconv", "mosaic_encoder_caps")) in graph.ds_pipeline.links
    assert (("mosaic_encoder_caps", "mosaic_force_idr")) in graph.ds_pipeline.links
    assert (("mosaic_force_idr", "mosaic_h264_encoder")) in graph.ds_pipeline.links
    assert (("mosaic_h264_encoder", "mosaic_h264_parse")) in graph.ds_pipeline.links
    assert (("mosaic_h264_parse", "mosaic_h264_au_caps")) in graph.ds_pipeline.links
    assert (("mosaic_h264_au_caps", "mosaic_webrtc_au_queue")) in graph.ds_pipeline.links
    assert (("mosaic_webrtc_au_queue", "mosaic_h264_shmsink")) in graph.ds_pipeline.links
    assert (("streammux", "orderly_eos_control")) in graph.ds_pipeline.links
    assert (("orderly_eos_control", "preprocess")) in graph.ds_pipeline.links
    assert (("yolo11_pgie", "main_tee")) in graph.ds_pipeline.links
    assert graph.components["mosaic_force_idr"].element == "noesisforceidr"
    assert graph.components["mosaic_h264_encoder"].element == "nvv4l2h264enc"
    assert graph.components["mosaic_h264_encoder"].config["profile"] == 0
    assert graph.components["mosaic_h264_encoder"].config["bitrate"] == 12_000_000
    assert graph.components["mosaic_h264_encoder"].config["iframeinterval"] == 10
    assert graph.components["mosaic_h264_encoder"].config["idrinterval"] == 10
    assert graph.components["mosaic_h264_encoder"].config["insert-sps-pps"] is True
    assert graph.components["mosaic_h264_parse"].config["config-interval"] == -1
    assert graph.components["mosaic_encode_queue"].config["leaky"] == 2
    assert graph.components["mosaic_webrtc_au_queue"].config["leaky"] == 0
    assert graph.components["mosaic_h264_shmsink"].config["wait-for-connection"] is False
    assert "rtsp_out" not in graph.components
    for component in graph.components.values():
        if component.element == "valve":
            assert int(component.config["drop-mode"]) in (1, 2)

    assert pipeline.prepare() is True
    assert pipeline.activate() is True


def test_detection_wake_performance_configs_are_bounded():
    tracker_cfg = yaml.safe_load((ROOT / "config" / "nvtracker.yaml").read_text(encoding="utf-8"))
    assert int(tracker_cfg["TargetManagement"]["maxTargetsPerStream"]) <= 64
    assert int(tracker_cfg["TargetManagement"]["maxShadowTrackingAge"]) <= 300
    assert int(tracker_cfg["VisualTracker"]["useHog"]) == 0
    assert int(tracker_cfg["VisualTracker"]["featureImgSizeLevel"]) <= 3
    assert int(tracker_cfg["ReID"]["reidType"]) == 2
    assert int(tracker_cfg["ReID"]["reidHistorySize"]) <= 128

    reid_ini = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False)
    reid_ini.optionxform = str
    reid_ini.read(ROOT / "pipelines" / "config_infer_secondary_reid_swin.ini", encoding="utf-8")
    reid_props = reid_ini["property"]
    assert int(reid_props["secondary-reinfer-interval"]) >= 12
    assert int(reid_props["classifier-async-mode"]) == 0

    pose_ini = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False)
    pose_ini.optionxform = str
    pose_ini.read(ROOT / "pipelines" / "config_infer_secondary_yolo26_pose.ini", encoding="utf-8")
    pose_props = pose_ini["property"]
    assert int(pose_props["secondary-reinfer-interval"]) >= 8
    assert int(pose_props["classifier-async-mode"]) == 0


def test_optional_rtsp_tooling_branch_is_nonleaky_after_encode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NOESIS_MOSAIC_RTSP_ENABLED", "1")
    graph = pipeline.build_pipeline(_infer_config_path())

    assert graph.components["mosaic_rtsp_out_queue"].config["leaky"] == 0
    assert graph.components["mosaic_rtsp_out_queue"].config["max-size-buffers"] == 4
    assert graph.components["rtsp_out"].config["bypass-codecs"] is True
    assert (("mosaic_h264_tee", "mosaic_rtsp_out_queue")) in graph.ds_pipeline.links
    assert (("mosaic_rtsp_out_queue", "rtsp_out")) in graph.ds_pipeline.links


def test_build_pipeline_honors_analytics_exclude_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    exclude_path = tmp_path / "config_nvdsanalytics_exclude.ini"
    monkeypatch.setenv("NOESIS_ANALYTICS_EXCLUDE_CONFIG", str(exclude_path))

    graph = pipeline.build_pipeline(_infer_config_path())

    assert graph.components["analytics_exclude"].config["config-file"] == str(exclude_path)


def test_exclusion_stream_coverage_accepts_exact_disabled_entries():
    pipeline.validate_exclusion_stream_coverage(
        [{}, {}, {}],
        {
            "stages": {
                "exclude": {
                    "streams": {
                        "0": {"roi_filtering": {"enable": False, "rois": []}},
                        "1": {"roi_filtering": {"enable": True, "rois": [{}]}},
                        "2": {"roi_filtering": {"enable": False, "rois": []}},
                    }
                }
            }
        },
    )


def test_exclusion_element_is_exact_and_has_no_substitute():
    assert (
        pipeline.require_native_exclusion_element({"element": "nvdsroiexclude"})
        == "nvdsroiexclude"
    )
    for config in ({}, {"element": "nvdsanalytics"}, {"element": ""}):
        with pytest.raises(RuntimeError, match="requires element='nvdsroiexclude'"):
            pipeline.require_native_exclusion_element(config)


@pytest.mark.parametrize(
    ("sources", "stream_ids", "message"),
    [
        ([{}, {}, {}], ["0", "1"], "missing"),
        ([{}, {}], ["0", "1", "2"], "extra"),
        ([{}, {}], ["0", "01"], "non-canonical"),
        ([{}, {"source-id": 7}], ["0", "1"], "canonical and contiguous"),
    ],
)
def test_exclusion_stream_coverage_rejects_source_policy_drift(
    sources, stream_ids, message: str
):
    analytics = {
        "stages": {
            "exclude": {
                "streams": {stream_id: {} for stream_id in stream_ids},
            }
        }
    }

    with pytest.raises(RuntimeError, match=message):
        pipeline.validate_exclusion_stream_coverage(sources, analytics)


def test_ds9_build_pipeline_honors_analytics_exclude_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exclude_path = tmp_path / "ds9_config_nvdsanalytics_exclude.ini"
    module_name = "ds9_pipeline_analytics_override_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "DS9" / "noesis" / "pipelines" / "ds8_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    ds9_pipeline = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = ds9_pipeline
    spec.loader.exec_module(ds9_pipeline)
    monkeypatch.setattr(
        ds9_pipeline,
        "materialize_nvinfer_engine_only_config",
        lambda **kwargs: Path(kwargs["source_config"]),
    )
    monkeypatch.setattr(
        ds9_pipeline,
        "materialize_nvtracker_engine_only_config",
        lambda **kwargs: Path(kwargs["source_config"]),
    )
    monkeypatch.setenv("NOESIS_ANALYTICS_EXCLUDE_CONFIG", str(exclude_path))

    graph = ds9_pipeline.build_pipeline(ROOT / "DS9" / "config" / "infer.yaml")

    assert graph.components["analytics_exclude"].config["config-file"] == str(exclude_path)
    assert graph.components["osd"].config["process-mode"] == 1
    assert graph.components["depth_tracking_queue"].config == {
        "leaky": 2,
        "max-size-buffers": 2,
        "max-size-bytes": 0,
        "max-size-time": 0,
    }
    assert graph.components["streammux"].config["buffer-pool-size"] == 8
    assert graph.components["tiler"].config["buffer-pool-size"] == 8
    expected_latest_only_queue = {
        "leaky": 2,
        "max-size-buffers": 4,
        "max-size-bytes": 0,
        "max-size-time": 0,
    }
    for source_id in range(3):
        queue_name = f"source_decode_queue_{source_id}"
        assert graph.components[queue_name].config == expected_latest_only_queue
        assert (f"source_{source_id}", queue_name) in graph.ds_pipeline.links
        assert (queue_name, f"dewarper_conv_{source_id}") in graph.ds_pipeline.links
    assert graph.source_progress_targets == {
        0: "dewarper_caps_out_0",
        1: "dewarper_caps_out_1",
        2: "dewarper_caps_out_2",
    }
    assert graph.source_progress_monitor is not None
    rgb_convert = graph.components["mapanything_rgb_convert"]
    rgb_caps = graph.components["mapanything_rgb_caps"]
    assert rgb_convert.element == "nvvideoconvert"
    assert rgb_convert.config["gpu-id"] == graph.components["streammux"].config[
        "gpu-id"
    ]
    assert rgb_convert.config["nvbuf-memory-type"] == graph.components[
        "streammux"
    ].config["nvbuf-memory-type"]
    assert rgb_caps.element == "capsfilter"
    assert "format=RGB," in rgb_caps.config["caps"]
    assert "format=RGBA" not in rgb_caps.config["caps"]
    assert graph.components["mapanything_queue"].downstream == [
        "mapanything_valve"
    ]
    assert graph.components["mapanything_valve"].downstream == [
        "mapanything_fullframe"
    ]
    assert graph.components["mapanything_fullframe"].downstream == [
        "mapanything_rgb_convert"
    ]
    assert rgb_convert.downstream == ["mapanything_rgb_caps"]
    assert rgb_caps.downstream == ["mapanything_fullframe_sink"]
    assert ("mapanything_valve", "mapanything_fullframe") in graph.ds_pipeline.links
    assert ("mapanything_fullframe", "mapanything_rgb_convert") in graph.ds_pipeline.links
    assert ("mapanything_rgb_convert", "mapanything_rgb_caps") in graph.ds_pipeline.links
    assert ("mapanything_rgb_caps", "mapanything_fullframe_sink") in graph.ds_pipeline.links


def test_enable_depth_controls_valve_and_timer(monkeypatch: pytest.MonkeyPatch):
    config_path = _infer_config_path()
    graph = pipeline.build_pipeline(config_path)
    assert graph.depth_enabled is False

    timers: List["DummyTimer"] = []

    class DummyTimer:
        def __init__(self, interval: float, callback: Callable[[], None]):
            self.interval = interval
            self.callback = callback
            self.cancelled = False
            self.started = False
            self.daemon = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

    monkeypatch.setattr(pipeline.threading, "Timer", lambda interval, func: DummyTimer(interval, func))

    payload = pipeline.enable_depth(seconds=3)
    assert payload["enabled"] is True
    assert payload["seconds"] == 3
    assert payload["will_disable_at"] >= payload["started_at"]
    assert len(timers) == 1

    valve = graph.components["mapanything_valve"]
    assert valve.config["drop"] is False
    assert graph.depth_enabled is True
    first_timer = timers[0]
    assert first_timer.started is True

    first_timer.callback()
    assert graph.depth_enabled is False
    assert valve.config["drop"] is True

    second_payload = pipeline.enable_depth(seconds=1)
    assert second_payload["enabled"] is True
    assert first_timer.cancelled is True
    assert len(timers) == 2
    assert graph.depth_enabled is True
    assert valve.config["drop"] is False


def test_depth_fps_zero_when_disabled():
    config_path = _infer_config_path()
    graph = pipeline.build_pipeline(config_path)

    assert graph.depth_fps(window=1.0) == 0.0

    graph.mark_depth_enabled(True)
    graph.record_depth_frame()
    graph.record_depth_frame()
    fps_enabled = graph.depth_fps(window=2.0)
    assert fps_enabled > 0.0

    graph.mark_depth_enabled(False)
    graph.record_depth_frame()
    fps_disabled = graph.depth_fps(window=2.0)
    assert fps_disabled == pytest.approx(0.0)
    assert graph.depth_frame_samples == []
