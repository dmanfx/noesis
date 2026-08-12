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
    parser.read(ROOT / "pipelines" / "config_infer_primary_yolo11_seg.ini", encoding="utf-8")

    assert parser.getint("class-attrs-all", "topk") == 30

    source = (ROOT / "pipelines" / "nvdsinfer_yolo11_seg" / "nvdsinfer_yolo11_seg.cpp").read_text(
        encoding="utf-8"
    )
    assert "NOESIS_YOLO11_SEG_PARSER_TOPK" in source
    assert "std::stable_sort" in source


def test_v3dt_ds9_adapter_is_parity_but_remains_dynamically_evidence_gated() -> None:
    matrix = yaml.safe_load((ROOT / "DS9" / "docs" / "runtime_ownership.yaml").read_text(encoding="utf-8"))
    capability = next(item for item in matrix["capabilities"] if item["id"] == "tracking.v3dt")
    assert capability["status"] == "parity"
    assert capability["acceptance"]["owner"] == "ds9-v3dt-adapter"
    reason = capability["acceptance"]["reason"]
    assert "same-session v2 global-world" in reason
    exit_criteria = capability["acceptance"]["exit_criteria"]
    assert "MV3DT overlap fusion" in exit_criteria
    evidence = capability["evidence"]["repository_source"]
    assert evidence["ds9_config"]["path"] == "DS9/config/infer_v3dt.yaml"
    assert evidence["ds9_assets"]["path"] == "DS9/noesis/v3dt_assets.py"
    assert not (ROOT / "config" / "infer_v3dt_ds9.yaml").exists()


def test_ds9_runtime_has_ds8_cli_compatibility_without_ds8_runtime_imports() -> None:
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


def test_ds8_ds9_floor_ray_admission_gate_is_exact_parity() -> None:
    method = "_admit_floor_ray_range"
    class_name = "_AnalyticsTelemetryProcessor"
    assert _method_ast(
        ROOT / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    ) == _method_ast(
        ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    )


def test_ds8_ds9_track_image_size_authority_is_exact_parity() -> None:
    class_name = "_AnalyticsTelemetryProcessor"
    method = "_intrinsics_base_image_size"
    assert _method_ast(
        ROOT / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    ) == _method_ast(
        ROOT / "DS9" / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    )


def test_ds8_protected_v3dt_track_image_size_authority_is_exact_parity() -> None:
    class_name = "_AnalyticsTelemetryProcessor"
    method = "_intrinsics_base_image_size"
    assert _method_ast(
        ROOT / "noesis" / "pipelines" / "hooks.py",
        class_name,
        method,
    ) == _method_ast(
        ROOT / "noesis" / "pipelines" / "hooks_v3dt_reimpl.py",
        class_name,
        method,
    )


def test_runtime_media_readiness_no_longer_owns_an_rtsp_ingress_probe() -> None:
    runtime_paths = (
        ROOT / "noesis" / "ds8_runtime.py",
        ROOT / "noesis" / "ds8_runtime_v3dt_reimpl.py",
        ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
    )
    for path in runtime_paths:
        source = path.read_text(encoding="utf-8")
        assert "def _probe_rtsp_describe" not in source
        assert "def _wait_for_rtsp_ready" not in source
        assert "MosaicH264ShmFeeder ready:" not in source


def test_v3dt_webrtc_startup_matches_ds8_and_ds9_shm_fanout_contract() -> None:
    runtime_paths = (
        ROOT / "noesis" / "ds8_runtime.py",
        ROOT / "noesis" / "ds8_runtime_v3dt_reimpl.py",
        ROOT / "DS9" / "noesis" / "ds9_runtime_core.py",
    )
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

    v3dt_source = runtime_paths[1].read_text(encoding="utf-8")
    assert "for slot in range(max_webrtc_clients):" not in v3dt_source
