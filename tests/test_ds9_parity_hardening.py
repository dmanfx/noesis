from __future__ import annotations

import ast
import configparser
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _method_ast(path: Path, class_name: str, method_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return ast.dump(child, include_attributes=False)
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def test_yolo11_seg_has_bounded_mask_topk() -> None:
    parser = configparser.ConfigParser()
    parser.read(
        ROOT / "DS9" / "pipelines" / "config_infer_primary_yolo11_seg.ini",
        encoding="utf-8",
    )

    assert 0 < parser.getint("class-attrs-all", "topk") <= 300

    source = (ROOT / "DS9" / "pipelines" / "nvdsinfer_yolo11_seg" / "nvdsinfer_yolo11_seg.cpp").read_text(
        encoding="utf-8"
    )
    assert "det.perClassPreclusterThreshold" in source
    assert "CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE" in source


def test_v3dt_ds9_adapter_remains_explicitly_opt_in() -> None:
    matrix = yaml.safe_load((ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(encoding="utf-8"))
    capability = next(item for item in matrix["capabilities"] if item["id"] == "mv3dt_kitchen_family")
    assert capability["status"] == "opt_in"
    assert capability["owner_path"] == "DS9/config/infer_mv3dt.yaml"
    assert not (ROOT / "config" / "infer_v3dt_ds9.yaml").exists()


def test_ds9_runtime_retains_supported_profile_cli_aliases_without_retired_imports() -> None:
    source = (ROOT / "DS9" / "noesis" / "ds9_runtime_core.py").read_text(encoding="utf-8")
    assert '"--pgie-profile"' in source
    assert '"--pgie_profile"' in source
    assert '"-pgie-profile"' in source
    assert '"--size"' in source
    executable_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "DS9" / "noesis" / "ds9_runtime.py",
            ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
            ROOT / "DS9" / "noesis" / "runtime_config.py",
        )
    )
    assert "noesis.ds8_runtime" not in executable_sources
    assert "noesis.ds8_preflight" not in executable_sources


def test_canonical_hooks_own_floor_ray_admission_gate() -> None:
    method = "_admit_floor_ray_range"
    class_name = "_AnalyticsTelemetryProcessor"
    assert _method_ast(
        ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    )


def test_canonical_hooks_own_track_image_size_authority() -> None:
    class_name = "_AnalyticsTelemetryProcessor"
    method = "_intrinsics_base_image_size"
    assert _method_ast(
        ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    )


def test_runtime_media_readiness_no_longer_owns_an_rtsp_ingress_probe() -> None:
    runtime_paths = (ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",)
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        assert "def _probe_rtsp_describe" not in source
        assert "def _wait_for_rtsp_ready" not in source
        assert "MosaicH264ShmFeeder ready:" not in source


def test_ds9_webrtc_startup_uses_shm_fanout_contract() -> None:
    runtime_paths = (ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",)
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        assert "MosaicH264ShmFeeder" in source
        assert "h264_feeder=mosaic_h264_feeder" in source
        assert "on_fatal_error=_mosaic_transport_failed" in source
        assert "path=rtsp_path" not in source
        assert "NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS" in source
        assert "register_webrtc_gateway_factory" in source
        assert "for slot in range(initial_webrtc_clients):" in source
        assert "WebRTC gateway capacity: %d max, %d warm slot(s)" in source
        assert "webrtc_gateways = []" not in source
        assert "list(webrtc_gateways) + list(detached_gateways)" not in source

    v3dt_source = runtime_paths[0].read_text(encoding="utf-8")
    assert "for slot in range(max_webrtc_clients):" not in v3dt_source
