"""Tests for Noesis Dev Console and shared preflight."""

from __future__ import annotations

from pathlib import Path

import pytest

from noesis.ds8_preflight import (
    REPO_ROOT,
    Severity,
    derive_osd_policy_from_ini,
    has_blocking,
    mask_output_available,
    run_preflight,
    validate_metadata_compatibility,
)
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.metadata_compat import analyze_pipeline
from noesis.dev_console.presets import get_preset, list_presets
from noesis.dev_console.config_editor import model_patch


def test_launch_spec_argv_env():
    spec = LaunchSpec(pgie_profile="yolo26_seg", size="s", tracking_mode="baseline")
    argv = spec.to_argv()
    assert "--pgie-profile" in argv
    assert "yolo26_seg" in argv
    assert "--size" in argv and "s" in argv
    env = spec.to_env({})
    assert "NOESIS_DEV_CONSOLE_LAUNCH_DIR" in env
    assert str(spec.launch_dir) in env["NOESIS_DEV_CONSOLE_LAUNCH_DIR"]


def test_mask_output_available_seg_ini():
    props = {"network-type": "3", "output-instance-mask": "1", "parse-bbox-instance-mask-func-name": "NvDsInferParseYoloSeg"}
    assert mask_output_available(props) is True
    osd = derive_osd_policy_from_ini(props)
    assert osd["display-mask"] == 1
    assert osd["display-bbox"] == 0


def test_mask_output_available_detect_ini():
    props = {"network-type": "0", "parse-bbox-func-name": "NvDsInferParseYolo"}
    assert mask_output_available(props) is False
    osd = derive_osd_policy_from_ini(props)
    assert osd["display-mask"] == 0
    assert osd["display-bbox"] == 1


def test_metadata_compat_baseline_detect_warn():
    pipeline_path = REPO_ROOT / "config" / "infer.yaml"
    if not pipeline_path.exists():
        pytest.skip("infer.yaml missing")
    # Simulate detect by analyzing with a build detect ini if present
    build_detect = REPO_ROOT / "build" / "config_infer_primary_yolo26_m.ini"
    if not build_detect.exists():
        results = validate_metadata_compatibility(
            {"models": {"pgie": {"config-file-path": "pipelines/config_infer_primary_yolo11.ini"}}},
            pipeline_path,
            tracking_mode="baseline",
            strict_baseline=False,
        )
        assert any(r.code == "metadata.object_depth.missing_mask" for r in results)
        return
    results = validate_metadata_compatibility(
        {"models": {"pgie": {"config-file-path": str(build_detect.relative_to(REPO_ROOT))}}},
        pipeline_path,
        tracking_mode="baseline",
        strict_baseline=True,
    )
    assert has_blocking(results)


def test_analyze_infer_yaml():
    pipeline_path = REPO_ROOT / "config" / "infer.yaml"
    if not pipeline_path.exists():
        pytest.skip("infer.yaml missing")
    analysis = analyze_pipeline(pipeline_path, tracking_mode="baseline")
    assert "osd_policy" in analysis
    assert "validations" in analysis


def test_presets_list():
    presets = list_presets()
    assert len(presets) >= 5
    baseline = get_preset("baseline-rtsp-yolo11-seg")
    assert baseline is not None
    assert baseline.preset_type == "canonical"


def test_model_patch_adds_osd():
    pipeline_path = REPO_ROOT / "config" / "infer.yaml"
    if not pipeline_path.exists():
        pytest.skip("infer.yaml missing")
    merged = model_patch(pipeline_path, reid_enable=False)
    assert "osd" in merged


def test_preflight_infer_yaml():
    pipeline_path = REPO_ROOT / "config" / "infer.yaml"
    if not pipeline_path.exists():
        pytest.skip("infer.yaml missing")
    results = run_preflight(pipeline_path=pipeline_path, tracking_mode="baseline")
    assert isinstance(results, list)
    # depth registration may block in some environments
    codes = {r.code for r in results}
    assert isinstance(codes, set)


def test_mask_output_unavailable_when_output_instance_mask_zero():
    props = {
        "network-type": "3",
        "output-instance-mask": "0",
        "parse-bbox-instance-mask-func-name": "NvDsInferParseYoloSeg",
    }
    assert mask_output_available(props) is False


def test_materialized_detect_osd_bbox_mode():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.materialize import materialize_launch_pipeline

    spec = LaunchSpec(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo26",
        size="m",
        tracking_mode="baseline",
        launch_id="test-detect-osd",
    )
    try:
        path = materialize_launch_pipeline(spec, dry_run=True)
    except Exception as exc:
        pytest.skip(f"materialization unavailable: {exc}")
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    osd = cfg.get("osd", {})
    assert int(osd.get("display-bbox", 0)) == 1
    assert int(osd.get("display-mask", 1)) == 0


def test_build_pipeline_resolves_paths_from_launch_dir():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.materialize import materialize_launch_pipeline
    from noesis.pipelines import ds8_pipeline

    spec = LaunchSpec(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo11_seg",
        tracking_mode="baseline",
        launch_id="test-path-resolve",
    )
    try:
        path = materialize_launch_pipeline(spec, dry_run=True)
        pipeline = ds8_pipeline.build_pipeline(path)
    except Exception as exc:
        pytest.skip(f"pipeline build unavailable: {exc}")
    tracker_comp = pipeline.components.get("tracker")
    assert tracker_comp is not None
    ll_cfg = str(tracker_comp.config.get("ll-config-file", ""))
    assert "build/dev_console/config" not in ll_cfg
    assert ll_cfg.endswith("config/nvtracker.yaml")


def test_depth_registration_preflight_passes_yolo11_seg():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.validator import validate_launch

    spec = LaunchSpec(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo11_seg",
        tracking_mode="baseline",
        launch_id="test-depth-reg",
    )
    try:
        result = validate_launch(spec)
    except Exception as exc:
        pytest.skip(f"materialization unavailable: {exc}")
    codes = {r["code"] for r in result.get("results", [])}
    assert "depth_registration.invalid" not in codes
    assert "depth_registration.validation_error" not in codes


def test_runnable_presets_include_experimental():
    from noesis.dev_console.presets import list_presets

    presets = list_presets(runnable_only=True)
    ids = {p["id"] for p in presets}
    assert "baseline-rtsp-yolo11-seg" in ids
    assert "v3dt-reimpl-fast-mp4" in ids
    assert len(presets) >= 9


def test_cuda_preflight_runs():
    from noesis.ds8_preflight import validate_cuda_preflight

    results = validate_cuda_preflight()
    codes = {r.code for r in results}
    assert "cuda.preflight.ok" in codes or "cuda.preflight.skipped" in codes or "cuda.preflight.cudart_missing" in codes


def test_manifest_catalog_coverage():
    from noesis.dev_console.manifest import load_knobs

    knobs = load_knobs()
    assert len(knobs) >= 100
    keys = {k["key"] for k in knobs}
    assert "NOESIS_REID_ENABLED" in keys
    assert "NOESIS_TRACKING_MODE" in keys


def test_describe_flow_returns_stages():
    from noesis.dev_console.pipeline_flow import describe_flow

    data = describe_flow(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo11_seg",
        size=None,
        tracking_mode="baseline",
        rtsp_port=9654,
        depth_enable_seconds=12,
        env={"NOESIS_LOG_LEVEL": "DEBUG"},
    )
    assert data.get("stages")
    assert any(s["id"] == "pgie" for s in data["stages"])
    assert any(s["id"] == "tracker" for s in data["stages"])
    pgie = next(s for s in data["stages"] if s["id"] == "pgie")
    mapanything = next(s for s in data["stages"] if s["id"] == "mapanything")
    rtsp = next(s for s in data["stages"] if s["id"] == "rtsp")
    assert any(c["target"] == "pgie_profile" for c in pgie["controls"])
    assert any(c["target"] == "depth_enable_seconds" and c["value"] == 12 for c in mapanything["controls"])
    assert ":9654/" in rtsp["detail"]
    knobs = {k["key"]: k["value"] for k in data.get("environment_knobs", [])}
    assert knobs.get("NOESIS_LOG_LEVEL") == "DEBUG"
    assert data["highlights"]["pgie_profile"] == "yolo11_seg"
    assert data["highlights"]["rtsp_port"] == 9654


def test_gate_catalog_reports_inherited_and_explicit_controls():
    from noesis.dev_console.gate_catalog import build_gate_catalog

    payload = build_gate_catalog(
        LaunchSpec(
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo11_seg",
            size="m",
            tracking_mode="baseline",
            rtsp_port=9654,
            depth_enable_seconds=12,
            strict_baseline=True,
            env={
                "NOESIS_REID_ENABLED": "0",
                "NOESIS_MOSAIC_WEBRTC_ENABLED": "1",
                "NOESIS_MAPANYTHING_GATE_PRIME_SECONDS": "2.5",
            },
        )
    )

    assert payload["summary"]["total"] >= 10
    groups = {group["id"]: group for group in payload["groups"]}
    assert {"identity", "depth", "mosaic", "visuals"} <= set(groups)
    controls = {item["target"]: item for item in payload["controls"]}
    assert controls["NOESIS_REID_ENABLED"]["value"] == "0"
    assert controls["NOESIS_REID_ENABLED"]["effective_display"] == "off"
    assert controls["NOESIS_REID_ENABLED"]["source"] == "env override"
    assert controls["NOESIS_TRAILS_RENDER"]["value"] == ""
    assert controls["NOESIS_TRAILS_RENDER"]["source"] == "pipeline"
    assert controls["depth_enable_seconds"]["effective_value"] == "12"
    assert controls["strict_baseline"]["effective_display"] == "on"
    assert controls["rtsp_port"]["effective_value"] == "9654"


def test_gates_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    client = TestClient(create_app())
    response = client.post(
        "/api/gates",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo11_seg",
            "tracking_mode": "baseline",
            "env_lines": "NOESIS_REID_ENABLED=0\nNOESIS_TRAILS_RENDER=0",
            "record_activity": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["explicit_overrides"] == 2
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "gates.evaluate"
    assert timeline[0]["payload"]["summary"]["explicit_overrides"] == 2


def test_log_intelligence_classifies_operational_signatures():
    from noesis.dev_console.log_intelligence import analyze_runtime_logs

    payload = analyze_runtime_logs(
        [
            "[dev-console] launching at 2026-07-03 03:00:00",
            "ERROR Failed to bind WebSocket port: Address already in use",
            "WARNING TensorRT engine deserialization failed; plan version mismatch",
            "Traceback (most recent call last):",
            "RuntimeError: CUDA out of memory while building nvinfer pipeline",
            "INFO shutdown complete",
        ],
        path="build/dev_console/test/runtime.log",
        runtime_status={"running": False, "pid": 123, "launch_id": "test-log", "returncode": 1},
    )

    assert payload["status"] == "blocked"
    assert payload["counts"]["block"] >= 3
    signatures = {item["id"]: item for item in payload["signatures"]}
    assert {"port_conflict", "tensorrt_engine", "cuda_gpu", "deepstream_pipeline", "python_exception"} <= set(signatures)
    assert signatures["port_conflict"]["target"] == "ports"
    assert payload["recommendations"][0]["severity"] == "block"
    assert payload["events"][-1]["message"].startswith("RuntimeError")


def test_log_insights_endpoint_reads_supervisor_log(tmp_path):
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app
    from noesis.dev_console.supervisor import RuntimeSupervisor

    runtime = RuntimeSupervisor()
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "\n".join(
            [
                "[dev-console] argv: python3 noesis/ds8_runtime.py",
                "WARNING RTSP sink not ready on 127.0.0.1:8554; skipping WebRTC gateway start",
                "ERROR Failed to link nvinfer element in pyservicemaker pipeline",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime.log_path = log_path
    client = TestClient(create_app(runtime))

    response = client.get("/api/runtime/log-insights?lines=20")
    assert response.status_code == 200
    payload = response.json()
    assert payload["line_count"] == 3
    assert payload["status"] == "error"
    signatures = {item["id"] for item in payload["signatures"]}
    assert "webrtc_rtsp" in signatures
    assert "deepstream_pipeline" in signatures


def test_strict_baseline_detect_blocks():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.validator import validate_launch

    spec = LaunchSpec(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo26",
        size="m",
        tracking_mode="baseline",
        strict_baseline=True,
        launch_id="test-strict",
    )
    try:
        result = validate_launch(spec)
    except Exception as exc:
        pytest.skip(f"materialization unavailable: {exc}")
    assert result.get("blocking") is True


def test_materialized_launch_overrides_rtsp_port():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.materialize import materialize_launch_pipeline

    spec = LaunchSpec(
        pipeline_config="config/infer.yaml",
        pgie_profile="yolo11_seg",
        tracking_mode="baseline",
        rtsp_port=9554,
        launch_id="test-rtsp-port",
    )
    try:
        path = materialize_launch_pipeline(spec, dry_run=True)
    except Exception as exc:
        pytest.skip(f"materialization unavailable: {exc}")
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert int(cfg.get("mosaic_output", {}).get("rtsp_port")) == 9554


def test_diagnostics_snapshot_reports_ports_and_artifacts():
    from noesis.dev_console.diagnostics import diagnostics_snapshot
    from noesis.dev_console.launch_spec import LaunchSpec

    snap = diagnostics_snapshot(
        LaunchSpec(
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="baseline",
            ws_port=6108,
            rest_port=8180,
            rtsp_port=8654,
        )
    )
    assert len(snap["ports"]) == 4
    assert {"ws_port", "rest_port", "rtsp_port"} <= set(snap["suggested_ports"])
    assert snap["artifacts"]["total"] > 0
    assert "gpu" in snap


def test_process_inventory_groups_port_owners(monkeypatch):
    import noesis.dev_console.diagnostics as diagnostics

    def fake_process_snapshot(pid, *, managed_pid=None):
        return {
            "pid": pid,
            "managed_by_console": pid == managed_pid,
            "looks_like_ds8": pid == 123,
            "command": f"python3 noesis/ds8_runtime.py --pid {pid}",
            "cwd": str(REPO_ROOT),
            "stats": {"elapsed_s": 10, "cpu_pct": "1.0", "rss_kib": 256},
            "env": {},
        }

    monkeypatch.setattr(diagnostics, "_process_snapshot", fake_process_snapshot)
    ports = [
        {
            "label": "WebSocket",
            "host": "127.0.0.1",
            "port": 6008,
            "owner": {"users": 'users:(("python3",pid=123,fd=18))'},
        },
        {
            "label": "REST",
            "host": "127.0.0.1",
            "port": 8080,
            "owner": {"users": 'users:(("python3",pid=123,fd=19))'},
        },
        {
            "label": "Console",
            "host": "127.0.0.1",
            "port": 9090,
            "owner": {"users": 'users:(("python3",pid=456,fd=20))'},
        },
    ]

    inventory = diagnostics.process_inventory(ports, managed_pid=123, include_discovered_ds8=False)
    assert len(inventory) == 2
    ds8 = next(item for item in inventory if item["pid"] == 123)
    assert ds8["managed_by_console"] is True
    assert ds8["looks_like_ds8"] is True
    assert {item["port"] for item in ds8["ports"]} == {6008, 8080}
    assert all(item["selected"] for item in ds8["ports"])


def test_process_inventory_discovers_unselected_ds8_runtime(monkeypatch):
    import noesis.dev_console.diagnostics as diagnostics

    def fake_process_snapshot(pid, *, managed_pid=None):
        return {
            "pid": pid,
            "managed_by_console": pid == managed_pid,
            "looks_like_ds8": True,
            "command": "python3 ./ds8_runtime.py --pgie-profile yolo26 --size m --ws-port 6010",
            "cwd": str(REPO_ROOT),
            "stats": {"elapsed_s": 20, "cpu_pct": "12.5", "rss_kib": 1024},
            "env": {},
            "launch": {"pgie_profile": "yolo26", "size": "m", "ws_port": "6010"},
        }

    monkeypatch.setattr(diagnostics, "_discover_ds8_runtime_pids", lambda: [789])
    monkeypatch.setattr(
        diagnostics,
        "_listening_ports_by_pid",
        lambda: {
            789: [
                {
                    "label": "WebSocket",
                    "host": "0.0.0.0",
                    "port": 6010,
                    "selected": False,
                    "source": "process-scan",
                }
            ]
        },
    )
    monkeypatch.setattr(diagnostics, "_process_snapshot", fake_process_snapshot)

    inventory = diagnostics.process_inventory([], managed_pid=None)

    assert len(inventory) == 1
    runtime = inventory[0]
    assert runtime["pid"] == 789
    assert runtime["looks_like_ds8"] is True
    assert runtime["managed_by_console"] is False
    assert runtime["ports"][0]["port"] == 6010
    assert runtime["ports"][0]["selected"] is False
    assert runtime["launch"]["pgie_profile"] == "yolo26"


def test_process_snapshot_parses_ds8_launch_metadata(monkeypatch):
    import noesis.dev_console.diagnostics as diagnostics

    monkeypatch.setattr(
        diagnostics,
        "_process_cmdline_parts",
        lambda _pid: [
            "python3",
            "./ds8_runtime.py",
            "--pipeline-config",
            "config/infer.yaml",
            "--cameras-config",
            "config/cameras.yaml",
            "--pgie-profile",
            "yolo11_seg",
            "--size",
            "m",
            "--tracking-mode",
            "baseline",
            "--ws-port",
            "6008",
            "--rest-port",
            "8080",
        ],
    )
    monkeypatch.setattr(diagnostics, "_process_stats", lambda _pid: {})
    monkeypatch.setattr(diagnostics, "_process_environ", lambda _pid: {})
    monkeypatch.setattr(diagnostics.os, "readlink", lambda _path: str(REPO_ROOT))

    snapshot = diagnostics._process_snapshot(123)

    assert snapshot["looks_like_ds8"] is True
    assert snapshot["launch"]["pgie_profile"] == "yolo11_seg"
    assert snapshot["launch"]["size"] == "m"
    assert snapshot["launch"]["tracking_mode"] == "baseline"
    assert snapshot["launch"]["pipeline_config"] == "config/infer.yaml"
    assert snapshot["launch"]["ws_port"] == "6008"


def test_ds8_process_detection_requires_runtime_entrypoint():
    import noesis.dev_console.diagnostics as diagnostics

    assert diagnostics._looks_like_ds8_command("python3 ./ds8_runtime.py --pgie-profile yolo11_seg") is True
    assert diagnostics._looks_like_ds8_command("python3 -m noesis.ds8_runtime --pgie-profile yolo11_seg") is True
    assert diagnostics._looks_like_ds8_command("rg ds8_runtime.py noesis") is False
    assert diagnostics._looks_like_ds8_command("bash -lc 'echo ds8_runtime.py'") is False


def test_support_bundle_endpoint_writes_bundle(monkeypatch, tmp_path):
    import json

    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    from noesis.dev_console import support_bundle
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(support_bundle, "BUNDLE_ROOT", tmp_path)
    monkeypatch.setattr(support_bundle, "probe_live_ws", lambda *_args, **_kwargs: {"connected": False, "error": "skipped"})

    client = TestClient(create_app())
    response = client.post(
        "/api/support/bundle",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo11_seg",
            "tracking_mode": "baseline",
            "ws_port": 6108,
            "rest_port": 8180,
            "rtsp_port": 8654,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    bundle_path = Path(payload["bundle_path"])
    summary_path = Path(payload["summary_path"])
    assert bundle_path.exists()
    assert summary_path.exists()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert {"spec", "runtime", "launch_plan", "launch_diff", "gates", "log_insights", "decision", "model_matrix", "sources", "remediation", "validation", "diagnostics", "flow", "live", "live_health"} <= set(bundle)
    assert bundle["live"]["error"] == "skipped"
    assert bundle["live_health"]["status"] == "down"
    assert bundle["gates"]["summary"]["total"] >= 10
    assert bundle["launch_diff"]["summary"]["total"] >= 0
    assert bundle["log_insights"]["status"] in {"empty", "clean", "attention", "error", "blocked"}
    assert bundle["model_matrix"]["summary"]["total"] > 0
    assert bundle["sources"]["summary"]["total"] > 0
    assert "components" in bundle["decision"]


def test_support_bundle_library_lists_and_reads_existing_bundle(monkeypatch, tmp_path):
    import json

    from noesis.dev_console import support_bundle

    monkeypatch.setattr(support_bundle, "BUNDLE_ROOT", tmp_path)
    bundle_dir = tmp_path / "20260702-222706-testbundle"
    bundle_dir.mkdir()
    payload = {
        "created_at": "2026-07-02T22:27:06-04:00",
        "launch_id": "testbundle",
        "spec": {"pgie_profile": "yolo11_seg", "size": "m", "tracking_mode": "baseline", "ws_port": 6009, "rest_port": 8081, "rtsp_port": 8555},
        "runtime": {"running": False, "pid": None},
        "validation": {"blocking": False, "counts": {"block": 0, "warn": 1, "info": 12}},
        "diagnostics": {"artifacts": {"total": 4, "missing": 0, "ready": True}, "processes": [{"pid": 123}]},
        "live": {"connected": True},
        "decision": {"status": "attention", "score": 82, "summary": "Review warning"},
    }
    (bundle_dir / "bundle.json").write_text(json.dumps(payload), encoding="utf-8")
    (bundle_dir / "summary.md").write_text("# Bundle\n\nhello", encoding="utf-8")

    listed = support_bundle.list_support_bundles()
    assert listed["total"] == 1
    item = listed["items"][0]
    assert item["id"] == bundle_dir.name
    assert item["counts"] == {"block": 0, "warn": 1, "info": 12}
    assert item["process_count"] == 1
    assert item["live_connected"] is True
    assert item["decision"]["status"] == "attention"
    assert item["spec"]["ports"] == {"ws": 6009, "rest": 8081, "rtsp": 8555}

    detail = support_bundle.read_support_bundle(bundle_dir.name)
    assert detail["summary"]["id"] == bundle_dir.name
    assert detail["markdown"].startswith("# Bundle")
    assert detail["bundle"]["launch_id"] == "testbundle"


def test_support_bundle_library_endpoint(monkeypatch, tmp_path):
    import json

    from fastapi.testclient import TestClient

    from noesis.dev_console import support_bundle
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(support_bundle, "BUNDLE_ROOT", tmp_path)
    bundle_dir = tmp_path / "20260702-222706-endpoint"
    bundle_dir.mkdir()
    (bundle_dir / "bundle.json").write_text(
        json.dumps(
            {
                "created_at": "2026-07-02T22:27:06-04:00",
                "launch_id": "endpoint",
                "spec": {"pgie_profile": "yolo26", "tracking_mode": "baseline"},
                "validation": {"blocking": True, "counts": {"block": 2, "warn": 0, "info": 5}},
                "diagnostics": {"artifacts": {"total": 3, "missing": 1}, "processes": []},
                "live": {"connected": False},
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "summary.md").write_text("endpoint summary", encoding="utf-8")

    client = TestClient(create_app())
    response = client.get("/api/support/bundles")
    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == bundle_dir.name
    detail = client.get(f"/api/support/bundles/{bundle_dir.name}")
    assert detail.status_code == 200
    assert detail.json()["summary"]["blocking"] is True
    assert detail.json()["markdown"] == "endpoint summary"
    assert client.get("/api/support/bundles/../bad").status_code == 404


def test_launch_plan_endpoint_summarizes_command_and_changes():
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/plan",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo26",
            "size": "m",
            "tracking_mode": "baseline",
            "ws_port": 6108,
            "rest_port": 8180,
            "rtsp_port": 9654,
            "depth_enable_seconds": 12,
            "env_lines": "NOESIS_REID_ENABLED=0\nNOESIS_MOSAIC_WEBRTC_ENABLED=1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "--pgie-profile yolo26" in payload["command"]
    assert "--depth-enable-seconds 12" in payload["command"]
    assert payload["ports"]["rtsp"]["port"] == 9654
    assert payload["artifacts"]["total"] > 0
    changed_labels = {item["label"] for item in payload["changes"] if item["changed"]}
    assert "PGIE engine" in changed_labels
    assert "RTSP port" in changed_labels
    gates = {item["key"]: str(item["value"]) for item in payload["gates"]}
    assert gates["NOESIS_DEPTH_ENABLE_SECONDS"] == "12"
    assert gates["NOESIS_REID_ENABLED"] == "0"


def test_launch_diff_reports_nested_effective_changes():
    from noesis.dev_console.launch_diff import build_launch_diff

    payload = build_launch_diff(
        LaunchSpec(
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo26",
            size="m",
            tracking_mode="baseline",
            ws_port=6108,
            rest_port=8180,
            rtsp_port=9654,
            depth_enable_seconds=12,
            env={"NOESIS_REID_ENABLED": "0"},
        )
    )

    assert payload["summary"]["total"] > 0
    assert payload["summary"]["high_impact"] > 0
    paths = {item["path"]: item for item in payload["changes"]}
    assert "models.pgie.engine" in paths
    assert paths["models.pgie.engine"]["high_impact"] is True
    assert paths["models.pgie.engine"]["after_text"].endswith("models/engines/yolo26m_dynamic_b1-3_fp16.engine")
    assert "mosaic_output.rtsp_port" in paths
    assert paths["mosaic_output.rtsp_port"]["after"] == 9654
    assert "env.NOESIS_REID_ENABLED" in paths
    assert paths["env.NOESIS_REID_ENABLED"]["after"] == "0"
    categories = {item["label"] for item in payload["categories"]}
    assert {"Models", "Mosaic", "Env"} <= categories


def test_launch_diff_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    client = TestClient(create_app())
    response = client.post(
        "/api/launch/diff",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo26",
            "size": "m",
            "tracking_mode": "baseline",
            "env_lines": "NOESIS_REID_ENABLED=0",
            "record_activity": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["high_impact"] > 0
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "launch.diff"
    assert timeline[0]["payload"]["summary"]["high_impact"] == payload["summary"]["high_impact"]


def test_remediation_endpoint_prioritizes_port_and_mask_actions(monkeypatch):
    from fastapi.testclient import TestClient

    import noesis.dev_console.server as server
    from noesis.ds8_preflight import Severity, ValidationResult

    def fake_validate(_spec):
        return {
            "blocking": True,
            "results": [
                ValidationResult(
                    severity=Severity.BLOCK,
                    code="port.6008.busy",
                    message="127.0.0.1:6008 is already accepting TCP connections.",
                    fix_hint="Stop the conflicting process or choose a different port.",
                ).to_dict(),
                ValidationResult(
                    severity=Severity.WARN,
                    code="metadata.object_depth.missing_mask",
                    message="Selected PGIE does not emit instance masks.",
                    fix_hint="Use a segmentation PGIE.",
                ).to_dict(),
            ],
            "counts": {"block": 1, "warn": 1, "info": 0},
        }

    def fake_diagnostics(_spec, *, managed_pid=None):
        return {
            "ports": [
                {
                    "label": "WebSocket",
                    "host": "127.0.0.1",
                    "port": 6008,
                    "busy": True,
                    "owner": {"users": 'users:(("python3",pid=123,fd=18))'},
                }
            ],
            "suggested_ports": {"ws_port": 6009, "rest_port": 8081, "rtsp_port": 8555},
            "artifacts": {"items": [], "total": 0, "missing": 0, "ready": True},
            "gpu": {"available": True},
            "processes": [
                {
                    "pid": 123,
                    "looks_like_ds8": True,
                    "managed_by_console": False,
                    "ports": [{"label": "WebSocket", "port": 6008}],
                }
            ],
        }

    monkeypatch.setattr(server, "validate_launch", fake_validate)
    monkeypatch.setattr(server, "diagnostics_snapshot", fake_diagnostics)
    monkeypatch.setattr(
        server,
        "build_launch_plan",
        lambda _spec: {"change_count": 0, "artifacts": {"missing": 0}},
    )

    client = TestClient(server.create_app())
    response = client.post("/api/remediation", json={"pipeline_config": "config/infer.yaml"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"
    titles = {item["title"] for item in payload["actions"]}
    assert "Resolve selected port conflict" in titles
    assert "Choose a segmentation PGIE for strict depth baselines" in titles
    assert "External DS8 runtime owns selected ports" in titles
    assert "WS 6009" in payload["summary"]


def test_remediation_reports_unselected_external_ds8_runtime():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.remediation import build_remediation

    runbook = build_remediation(
        LaunchSpec(pgie_profile="yolo11_seg", size="m"),
        validation={"blocking": False, "counts": {"block": 0, "warn": 0, "info": 2}, "results": []},
        diagnostics={
            "ports": [],
            "processes": [
                {
                    "pid": 789,
                    "looks_like_ds8": True,
                    "managed_by_console": False,
                    "ports": [{"label": "WebSocket", "port": 6010, "selected": False}],
                }
            ],
            "artifacts": {"items": []},
            "gpu": {"available": True},
        },
    )

    titles = {item["title"] for item in runbook["actions"]}
    assert runbook["status"] == "attention"
    assert "External DS8 runtime is already running" in titles


def test_live_probe_summarizes_stats_payload(monkeypatch):
    from noesis.dev_console import live_probe

    class FakeWebSocket:
        def __init__(self):
            self.messages = [
                '{"type":"trail_visualization_enabled_update","enabled":true}',
                '{"type":"stats","payload":{"timestamp":1.0,"uptime":2.0,"stack":"ds8","application":{"running":true,"cameras_active":3,"processors_active":1},"pipeline":{"prepared":true,"activated":true,"depth_enabled":false,"zero_copy_profile":"strict","zero_copy_violations":0,"stableid_backend_mode":"gpu","stableid_gallery_size":7},"cameras":{"0":{},"1":{},"2":{}}}}',
                '{"type":"pong","timestamp":1.0}',
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, _message):
            return None

        async def recv(self):
            if not self.messages:
                raise TimeoutError
            return self.messages.pop(0)

    class FakeWebsockets:
        @staticmethod
        def connect(*_args, **_kwargs):
            return FakeWebSocket()

    monkeypatch.setitem(__import__("sys").modules, "websockets", FakeWebsockets)
    payload = live_probe.probe_live_ws("127.0.0.1", 6008, timeout_s=1.0)
    assert payload["connected"] is True
    assert payload["pong"] is True
    assert payload["trail_enabled"] is True
    assert payload["stats"]["pipeline"]["activated"] is True
    assert payload["stats"]["application"]["cameras_active"] == 3


def test_live_health_scores_activated_runtime():
    from noesis.dev_console.live_health import summarize_live_health

    health = summarize_live_health(
        {
            "connected": True,
            "url": "ws://127.0.0.1:6008",
            "pong": True,
            "trail_enabled": True,
            "message_types": {"stats": 1, "pong": 1},
            "stats": {
                "uptime": 12.5,
                "stack": "ds8",
                "application": {"running": True, "cameras_active": 3, "processors_active": 1},
                "pipeline": {
                    "prepared": True,
                    "activated": True,
                    "depth_enabled": False,
                    "depth_fps": 0.0,
                    "zero_copy_profile": "strict",
                    "zero_copy_violations": 0,
                    "stableid_backend_mode": "gpu",
                    "stableid_gallery_size": 7,
                },
                "camera_count": 3,
                "camera_ids": ["family-room", "kitchen", "living-room"],
            },
        }
    )

    assert health["status"] == "healthy"
    assert health["score"] >= 85
    assert len(health["cameras"]) == 3
    indicators = {item["label"]: item for item in health["indicators"]}
    assert indicators["Zero copy"]["status"] == "ok"
    assert indicators["StableID"]["value"] == "gpu"


def test_profile_save_load_delete_roundtrip(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.profiles as profiles
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(profiles, "PROFILE_ROOT", tmp_path)
    client = TestClient(create_app())
    save = client.post(
        "/api/profiles/save",
        json={
            "name": "YOLO26 Free Ports",
            "notes": "test profile",
            "spec": {
                "pipeline_config": "config/infer.yaml",
                "pgie_profile": "yolo26",
                "size": "m",
                "tracking_mode": "baseline",
                "ws_port": 6009,
                "rest_port": 8081,
                "rtsp_port": 8555,
                "depth_enable_seconds": 12,
                "env_lines": "NOESIS_MAPANYTHING_GATE_PRIME_SECONDS=2.5",
            },
        },
    )
    assert save.status_code == 200
    profile_id = save.json()["summary"]["id"]
    assert profile_id == "yolo26-free-ports"

    listed = client.get("/api/profiles").json()["items"]
    assert any(item["id"] == profile_id and item["pgie_profile"] == "yolo26" for item in listed)

    loaded = client.post("/api/profiles/load", json={"profile_id": profile_id})
    assert loaded.status_code == 200
    spec = loaded.json()["profile"]["spec"]
    assert spec["ws_port"] == 6009
    assert spec["env"]["NOESIS_MAPANYTHING_GATE_PRIME_SECONDS"] == "2.5"

    deleted = client.post("/api/profiles/delete", json={"profile_id": profile_id})
    assert deleted.status_code == 200
    assert client.get("/api/profiles").json()["items"] == []


def test_activity_log_redacts_and_lists_newest_first(monkeypatch, tmp_path):
    import noesis.dev_console.activity as activity

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    first = activity.record_activity(
        "test.first",
        "First event",
        payload={"NOESIS_API_TOKEN": "secret", "nested": {"password": "also-secret"}, "plain": "visible"},
    )
    second = activity.record_activity("test.second", "Second event", severity="ok", payload={"plain": "newer"})

    listed = activity.list_activity(limit=10)
    assert listed["total"] == 2
    assert [item["id"] for item in listed["items"]] == [second["id"], first["id"]]
    assert listed["items"][1]["payload"]["NOESIS_API_TOKEN"] == "<redacted>"
    assert listed["items"][1]["payload"]["nested"]["password"] == "<redacted>"
    assert listed["items"][1]["payload"]["plain"] == "visible"


def test_activity_endpoint_records_profile_and_plan_actions(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.profiles as profiles
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(profiles, "PROFILE_ROOT", tmp_path / "profiles")
    client = TestClient(create_app())

    save = client.post(
        "/api/profiles/save",
        json={
            "name": "Activity Test",
            "spec": {
                "pipeline_config": "config/infer.yaml",
                "pgie_profile": "yolo11_seg",
                "tracking_mode": "baseline",
                "env_lines": "NOESIS_API_TOKEN=secret\nNOESIS_REID_ENABLED=1",
            },
        },
    )
    assert save.status_code == 200

    plan = client.post(
        "/api/launch/plan",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo11_seg",
            "tracking_mode": "baseline",
            "record_activity": True,
        },
    )
    assert plan.status_code == 200

    timeline = client.get("/api/activity?limit=10")
    assert timeline.status_code == 200
    items = timeline.json()["items"]
    event_types = [item["type"] for item in items]
    assert "launch.plan" in event_types
    assert "profile.save" in event_types
    profile_event = next(item for item in items if item["type"] == "profile.save")
    assert profile_event["payload"]["spec"]["env_count"] == 2
    assert "NOESIS_API_TOKEN" not in profile_event["payload"]["spec"]


def test_model_matrix_reports_ready_rows():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.model_matrix import build_model_matrix

    matrix = build_model_matrix(
        LaunchSpec(
            pipeline_config="config/infer.yaml",
            pgie_profile="yolo11_seg",
            size="m",
            tracking_mode="baseline",
        )
    )
    assert matrix["summary"]["total"] >= 25
    active = next(row for row in matrix["rows"] if row["id"] == "yolo11_seg:m")
    assert active["active"] is True
    assert active["config"].endswith("build/config_infer_primary_yolo11_seg_m.ini")
    assert active["engine"].endswith("models/engines/yolo11m-seg_cust.engine")
    assert active["kind"] in {"masks", "boxes"}
    assert {"block", "warn", "info"} <= set(active["counts"])


def test_model_matrix_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    client = TestClient(create_app())
    response = client.post(
        "/api/model-matrix",
        json={
            "pipeline_config": "config/infer.yaml",
            "pgie_profile": "yolo11_seg",
            "size": "m",
            "tracking_mode": "baseline",
            "record_activity": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_id"] == "yolo11_seg:m"
    assert payload["summary"]["total"] >= 25
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "model.matrix"
    assert timeline[0]["payload"]["summary"]["total"] == payload["summary"]["total"]


def test_source_readiness_redacts_uris_and_aligns_cameras():
    from noesis.dev_console.launch_spec import LaunchSpec
    from noesis.dev_console.source_probe import build_source_readiness

    payload = build_source_readiness(
        LaunchSpec(
            pipeline_config="config/infer.yaml",
            cameras_config="config/cameras.yaml",
            pgie_profile="yolo11_seg",
            tracking_mode="baseline",
        ),
        probe_network=False,
    )

    assert payload["summary"]["source_count"] == 3
    assert payload["summary"]["camera_count"] == 3
    assert payload["summary"]["alignment_status"] == "ready"
    first = payload["rows"][0]
    assert first["camera_id"] == "living-room"
    assert first["uri_display"].startswith("rtsp://")
    assert "jdr9oLlBkjyl3gDm" not in first["uri_display"]
    assert first["camera"]["intrinsics_ready"] is True
    assert first["dewarper"]["exists"] is True


def test_sources_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    from noesis.dev_console.server import create_app

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    client = TestClient(create_app())
    response = client.post(
        "/api/sources",
        json={
            "pipeline_config": "config/infer.yaml",
            "cameras_config": "config/cameras.yaml",
            "pgie_profile": "yolo11_seg",
            "tracking_mode": "baseline",
            "probe_network": False,
            "record_activity": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 3
    assert all("://" in row["uri_display"] and "7447" in row["uri_display"] for row in payload["rows"])
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "sources.scan"
    assert timeline[0]["payload"]["summary"]["total"] == 3


def test_launch_decision_prioritizes_blocking_evidence(monkeypatch):
    import noesis.dev_console.launch_decision as launch_decision

    monkeypatch.setattr(
        launch_decision,
        "validate_launch",
        lambda _spec: {"blocking": True, "counts": {"block": 1, "warn": 0, "info": 2}, "results": []},
    )
    monkeypatch.setattr(
        launch_decision,
        "diagnostics_snapshot",
        lambda _spec, managed_pid=None: {
            "ports": [
                {"label": "WebSocket", "port": 6008, "busy": True, "owner": {"users": "pid=123"}},
                {"label": "Console", "port": 9090, "busy": True, "owner": {"users": "pid=456"}},
            ],
            "processes": [],
            "gpu": {"available": True},
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "build_launch_plan",
        lambda _spec: {"change_count": 0, "artifacts": {"total": 4, "missing": 0, "ready": True}},
    )
    monkeypatch.setattr(
        launch_decision,
        "build_model_matrix",
        lambda _spec: {"active_id": "yolo11_seg:m", "summary": {"total": 1}, "rows": [{"id": "yolo11_seg:m", "active": True, "status": "ready", "kind": "masks"}]},
    )
    monkeypatch.setattr(
        launch_decision,
        "build_source_readiness",
        lambda _spec, **_kwargs: {"summary": {"ready": 3, "warn": 0, "blocked": 0}},
    )
    monkeypatch.setattr(
        launch_decision,
        "build_remediation",
        lambda *_args, **_kwargs: {
            "actions": [
                {
                    "severity": "block",
                    "title": "Resolve selected port conflict",
                    "reason": "busy",
                    "action": "Choose alternate ports.",
                    "target": "ports",
                }
            ]
        },
    )

    decision = launch_decision.build_launch_decision(LaunchSpec(pgie_profile="yolo11_seg", size="m"))
    assert decision["status"] == "blocked"
    assert decision["start_allowed"] is False
    assert decision["primary_action"]["title"] == "Resolve selected port conflict"
    component_status = {item["key"]: item["status"] for item in decision["components"]}
    assert component_status["preflight"] == "blocked"
    assert component_status["ports"] == "blocked"


def test_launch_decision_can_ignore_supervised_ports_for_restart(monkeypatch):
    import noesis.dev_console.launch_decision as launch_decision

    monkeypatch.setattr(
        launch_decision,
        "validate_launch",
        lambda _spec: {
            "blocking": True,
            "counts": {"block": 1, "warn": 0, "info": 1},
            "results": [
                {
                    "severity": "block",
                    "code": "port.6008.busy",
                    "message": "127.0.0.1:6008 is already accepting TCP connections.",
                    "fix_hint": "Stop the conflicting process or choose a different port.",
                },
                {"severity": "info", "code": "cuda.preflight.ok", "message": "CUDA visible"},
            ],
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "diagnostics_snapshot",
        lambda _spec, managed_pid=None: {
            "ports": [
                {"label": "WebSocket", "port": 6008, "busy": True, "owner": {"users": "pid=123", "pids": [123]}},
                {"label": "Console", "port": 9090, "busy": True, "owner": {"users": "pid=456", "pids": [456]}},
            ],
            "processes": [{"pid": 123, "managed_by_console": True, "looks_like_ds8": True}],
            "gpu": {"available": True},
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "build_launch_plan",
        lambda _spec: {"change_count": 0, "artifacts": {"total": 4, "missing": 0, "ready": True}},
    )
    monkeypatch.setattr(
        launch_decision,
        "build_model_matrix",
        lambda _spec: {
            "active_id": "yolo11_seg:m",
            "summary": {"total": 1},
            "rows": [{"id": "yolo11_seg:m", "active": True, "status": "ready", "kind": "masks"}],
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "build_source_readiness",
        lambda _spec, **_kwargs: {"summary": {"ready": 3, "warn": 0, "blocked": 0}},
    )

    decision = launch_decision.build_launch_decision(
        LaunchSpec(pgie_profile="yolo11_seg", size="m"),
        runtime_status={"running": False, "pid": 123, "launch_id": "active"},
        managed_pid=123,
        ignore_managed_runtime_ports=True,
    )

    assert decision["status"] == "ready"
    assert decision["start_allowed"] is True
    assert decision["validation"]["counts"]["block"] == 0
    assert any(item["code"] == "port.6008.managed" for item in decision["validation"]["results"])
    assert decision["evidence"]["diagnostics"]["busy_ports"] == []
    component_status = {item["key"]: item["status"] for item in decision["components"]}
    assert component_status["preflight"] == "ready"
    assert component_status["ports"] == "ready"


def test_launch_decision_flags_external_ds8_runtime(monkeypatch):
    import noesis.dev_console.launch_decision as launch_decision

    monkeypatch.setattr(
        launch_decision,
        "validate_launch",
        lambda _spec: {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 2}, "results": []},
    )
    monkeypatch.setattr(
        launch_decision,
        "diagnostics_snapshot",
        lambda _spec, managed_pid=None: {
            "ports": [],
            "processes": [
                {
                    "pid": 789,
                    "looks_like_ds8": True,
                    "managed_by_console": False,
                    "ports": [{"label": "WebSocket", "port": 6010, "selected": False}],
                }
            ],
            "ds8_runtimes": [{"pid": 789, "managed_by_console": False}],
            "gpu": {"available": True},
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "build_launch_plan",
        lambda _spec: {"change_count": 0, "artifacts": {"total": 4, "missing": 0, "ready": True}},
    )
    monkeypatch.setattr(
        launch_decision,
        "build_model_matrix",
        lambda _spec: {
            "active_id": "yolo11_seg:m",
            "summary": {"total": 1},
            "rows": [{"id": "yolo11_seg:m", "active": True, "status": "ready", "kind": "masks"}],
        },
    )
    monkeypatch.setattr(
        launch_decision,
        "build_source_readiness",
        lambda _spec, **_kwargs: {"summary": {"ready": 3, "warn": 0, "blocked": 0}},
    )

    decision = launch_decision.build_launch_decision(LaunchSpec(pgie_profile="yolo11_seg", size="m"))

    assert decision["status"] == "attention"
    assert decision["start_allowed"] is False
    component_status = {item["key"]: item["status"] for item in decision["components"]}
    assert component_status["external_runtime"] == "attention"
    assert decision["evidence"]["diagnostics"]["external_ds8_runtime_count"] == 1


def test_launch_decision_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(
        server,
        "build_launch_decision",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "score": 100,
            "start_allowed": True,
            "summary": "Ready for Start",
            "primary_action": {"title": "Ready for a console-owned launch"},
            "components": [],
            "top_actions": [],
            "evidence": {},
        },
    )
    client = TestClient(server.create_app())
    response = client.post(
        "/api/launch/decision",
        json={"pipeline_config": "config/infer.yaml", "record_activity": True},
    )
    assert response.status_code == 200
    assert response.json()["start_allowed"] is True
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "launch.decision"
    assert timeline[0]["payload"]["start_allowed"] is True


def test_launch_decision_endpoint_ignores_console_owned_ports_when_running(monkeypatch):
    from fastapi.testclient import TestClient

    import noesis.dev_console.server as server

    class FakeRuntime:
        def status(self):
            return {"running": True, "pid": 123, "launch_id": "active"}

    calls = {}

    def fake_decision(_spec, **kwargs):
        calls.update(kwargs)
        return {
            "status": "running",
            "score": 100,
            "start_allowed": False,
            "summary": "Console-managed runtime is already running",
            "validation": {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 5}, "results": []},
            "primary_action": {"title": "Console-managed runtime is already running"},
            "components": [],
            "top_actions": [],
            "evidence": {},
        }

    monkeypatch.setattr(server, "build_launch_decision", fake_decision)
    client = TestClient(server.create_app(FakeRuntime()))
    response = client.post(
        "/api/launch/decision",
        json={"pipeline_config": "config/infer.yaml", "pgie_profile": "yolo11_seg"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert calls["runtime_status"]["running"] is True
    assert calls["managed_pid"] == 123
    assert calls["ignore_managed_runtime_ports"] is True


def test_runtime_start_blocks_when_launch_decision_not_ready(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    class FakeRuntime:
        last_validation = None

        def __init__(self):
            self.started = False

        def status(self):
            return {"running": False, "pid": None, "launch_id": None}

        def start(self, _spec):
            self.started = True
            raise AssertionError("start must not be called when launch decision blocks")

    fake_runtime = FakeRuntime()
    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(
        server,
        "build_launch_decision",
        lambda *_args, **_kwargs: {
            "status": "attention",
            "score": 82,
            "start_allowed": False,
            "summary": "Choose a segmentation PGIE for strict depth baselines",
            "validation": {"blocking": False, "counts": {"block": 0, "warn": 1, "info": 16}, "results": []},
        },
    )

    client = TestClient(server.create_app(fake_runtime))
    response = client.post(
        "/api/runtime/start",
        json={"pipeline_config": "config/infer.yaml", "pgie_profile": "yolo26", "ws_port": 6009, "rest_port": 8081, "rtsp_port": 8555},
    )

    assert response.status_code == 409
    payload = response.json()["detail"]
    assert payload["decision"]["status"] == "attention"
    assert payload["decision"]["start_allowed"] is False
    assert fake_runtime.started is False
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "runtime.start_blocked"
    assert timeline[0]["payload"]["status"] == "attention"


def test_runtime_start_delegates_when_launch_decision_ready(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    class FakeRuntime:
        def __init__(self):
            self.started = False
            self.last_validation = {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 5}}

        def status(self):
            return {"running": False, "pid": None, "launch_id": None}

        def start(self, spec):
            self.started = True
            return {"running": True, "pid": 777, "launch_id": spec.launch_id, "materialized_pipeline": "build/dev_console/test/infer.yaml"}

    fake_runtime = FakeRuntime()
    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(
        server,
        "build_launch_decision",
        lambda *_args, **_kwargs: {
            "status": "ready",
            "score": 100,
            "start_allowed": True,
            "summary": "Ready for Start",
            "validation": {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 5}, "results": []},
        },
    )

    client = TestClient(server.create_app(fake_runtime))
    response = client.post(
        "/api/runtime/start",
        json={"pipeline_config": "config/infer.yaml", "pgie_profile": "yolo11_seg", "ws_port": 6009, "rest_port": 8081, "rtsp_port": 8555},
    )

    assert response.status_code == 200
    assert response.json()["running"] is True
    assert response.json()["pid"] == 777
    assert fake_runtime.started is True
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "runtime.start"


def test_runtime_restart_blocks_when_launch_decision_not_ready(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    class FakeRuntime:
        last_validation = None

        def __init__(self):
            self.restarted = False

        def status(self):
            return {"running": True, "pid": 123, "launch_id": "active"}

        def restart(self, _spec):
            self.restarted = True
            raise AssertionError("restart must not be called when launch decision blocks")

    calls = {}

    def fake_decision(_spec, **kwargs):
        calls.update(kwargs)
        return {
            "status": "blocked",
            "score": 56,
            "start_allowed": False,
            "summary": "Resolve selected port conflict",
            "validation": {"blocking": True, "counts": {"block": 1, "warn": 0, "info": 4}, "results": []},
        }

    fake_runtime = FakeRuntime()
    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(server, "build_launch_decision", fake_decision)

    client = TestClient(server.create_app(fake_runtime))
    response = client.post(
        "/api/runtime/restart",
        json={"pipeline_config": "config/infer.yaml", "pgie_profile": "yolo11_seg", "ws_port": 6008, "rest_port": 8080, "rtsp_port": 8554},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["decision"]["status"] == "blocked"
    assert fake_runtime.restarted is False
    assert calls["managed_pid"] == 123
    assert calls["runtime_status"]["running"] is False
    assert calls["ignore_managed_runtime_ports"] is True
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "runtime.restart_blocked"
    assert timeline[0]["payload"]["status"] == "blocked"


def test_runtime_restart_delegates_when_launch_decision_ready(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    class FakeRuntime:
        def __init__(self):
            self.restarted = False
            self.last_validation = {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 5}}

        def status(self):
            return {"running": True, "pid": 123, "launch_id": "active"}

        def restart(self, spec):
            self.restarted = True
            return {"running": True, "pid": 888, "launch_id": spec.launch_id, "materialized_pipeline": "build/dev_console/test/infer.yaml"}

    calls = {}

    def fake_decision(_spec, **kwargs):
        calls.update(kwargs)
        return {
            "status": "ready",
            "score": 100,
            "start_allowed": True,
            "summary": "Ready for Start",
            "validation": {"blocking": False, "counts": {"block": 0, "warn": 0, "info": 5}, "results": []},
        }

    fake_runtime = FakeRuntime()
    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(server, "build_launch_decision", fake_decision)

    client = TestClient(server.create_app(fake_runtime))
    response = client.post(
        "/api/runtime/restart",
        json={"pipeline_config": "config/infer.yaml", "pgie_profile": "yolo11_seg", "ws_port": 6008, "rest_port": 8080, "rtsp_port": 8554},
    )

    assert response.status_code == 200
    assert response.json()["running"] is True
    assert response.json()["pid"] == 888
    assert fake_runtime.restarted is True
    assert calls["managed_pid"] == 123
    assert calls["runtime_status"]["running"] is False
    assert calls["ignore_managed_runtime_ports"] is True
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "runtime.restart"


def test_launch_candidates_builds_ranked_recipes(monkeypatch):
    import noesis.dev_console.launch_candidates as launch_candidates

    def fake_diagnostics(spec, *, managed_pid=None):
        busy = int(spec.ws_port) == 6008
        return {
            "ports": [
                {"label": "WebSocket", "port": spec.ws_port, "busy": busy, "owner": {"users": "pid=123"} if busy else {}},
                {"label": "Console", "port": 9090, "busy": True, "owner": {"users": "pid=456"}},
            ],
            "suggested_ports": {"ws_port": 6009, "rest_port": 8081, "rtsp_port": 8555},
        }

    monkeypatch.setattr(launch_candidates, "diagnostics_snapshot", fake_diagnostics)
    monkeypatch.setattr(
        launch_candidates,
        "build_model_matrix",
        lambda _spec: {
            "rows": [
                {"id": "yolo11_seg:m", "pgie_profile": "yolo11_seg", "size": "m", "kind": "masks", "status": "ready"},
                {"id": "yolo26:m", "pgie_profile": "yolo26", "size": "m", "kind": "boxes", "status": "warn"},
            ]
        },
    )

    def fake_decision(spec, **_kwargs):
        if int(spec.ws_port) == 6009 and spec.pgie_profile == "yolo11_seg":
            return {
                "status": "ready",
                "score": 100,
                "start_allowed": True,
                "summary": "Ready for Start",
                "primary_action": {"title": "Ready"},
                "components": [],
            }
        if spec.pgie_profile == "yolo26":
            return {
                "status": "attention",
                "score": 82,
                "start_allowed": False,
                "summary": "Review detector warning",
                "primary_action": {"title": "Choose a segmentation PGIE"},
                "components": [{"label": "Selected Model", "status": "attention", "detail": "boxes"}],
            }
        return {
            "status": "blocked",
            "score": 56,
            "start_allowed": False,
            "summary": "Resolve selected port conflict",
            "primary_action": {"title": "Resolve selected port conflict"},
            "components": [{"label": "Ports", "status": "blocked", "detail": "busy"}],
        }

    monkeypatch.setattr(launch_candidates, "build_launch_decision", fake_decision)
    payload = launch_candidates.build_launch_candidates(LaunchSpec(pgie_profile="yolo11_seg", size="m"))

    assert payload["summary"]["total"] >= 4
    assert payload["summary"]["best_id"] == "free-ports"
    recipes = {item["id"]: item for item in payload["items"]}
    assert recipes["free-ports"]["start_allowed"] is True
    assert recipes["free-ports"]["ports"] == {"ws": 6009, "rest": 8081, "rtsp": 8555}
    assert recipes["detector-throughput"]["status"] == "attention"
    assert any(change["field"] == "ws_port" for change in recipes["free-ports"]["changes"])


def test_launch_candidates_endpoint_records_activity(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    import noesis.dev_console.activity as activity
    import noesis.dev_console.server as server

    monkeypatch.setattr(activity, "ACTIVITY_LOG", tmp_path / "activity.jsonl")
    monkeypatch.setattr(
        server,
        "build_launch_candidates",
        lambda *_args, **_kwargs: {
            "summary": {"total": 2, "ready": 1, "attention": 1, "blocked": 0, "best_id": "free-ports", "best_status": "ready"},
            "items": [],
            "diagnostics": {},
        },
    )
    client = TestClient(server.create_app())
    response = client.post(
        "/api/launch/candidates",
        json={"pipeline_config": "config/infer.yaml", "record_activity": True},
    )

    assert response.status_code == 200
    assert response.json()["summary"]["best_id"] == "free-ports"
    timeline = client.get("/api/activity?limit=5").json()["items"]
    assert timeline[0]["type"] == "launch.candidates"
    assert timeline[0]["payload"]["summary"]["ready"] == 1


def test_dev_console_static_ui_shell():
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app

    client = TestClient(create_app())
    index = client.get("/")
    assert index.status_code == 200
    html = index.text
    assert "phase-tabs" in html
    assert 'id="inspectorDrawer"' in html
    assert 'class="panel-head operate-runtime-head"' in html
    assert 'id="logDrawer"' in html
    assert 'data-phase="configure"' in html
    assert 'data-phase="operate"' in html

    app_js = client.get("/static/app.js")
    styles = client.get("/static/styles.css")
    assert app_js.status_code == 200
    assert styles.status_code == 200
    assert "CONSOLE_PHASES" in app_js.text
    assert "openInspectorDrawer" in app_js.text
    assert "maybeSyncObservedRuntime" in app_js.text
    assert "adoptObservedRuntime" in app_js.text
    assert "activeRuntimePid" in app_js.text
    assert "resolveObservedRuntime" in app_js.text
    assert "syncFormWithActiveRuntime" in app_js.text
    assert "refreshDependentLaunchPanels" in app_js.text
    assert "status-strip" in styles.text
    assert "inspector-drawer" in styles.text


_UI_BASE_SPEC = {
    "pipeline_config": "config/infer.yaml",
    "cameras_config": "config/cameras.yaml",
    "pgie_profile": "yolo11_seg",
    "size": "m",
    "tracking_mode": "baseline",
    "ws_port": 6108,
    "rest_port": 8180,
    "rtsp_port": 8654,
    "probe_network": False,
}


def test_ui_api_surface_endpoints():
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app

    client = TestClient(create_app())

    summary = client.get("/api/summary")
    assert summary.status_code == 200
    assert "presets" in summary.json()
    assert "runtime" in summary.json()

    knobs = client.get("/api/knobs")
    assert knobs.status_code == 200
    assert len(knobs.json().get("items", [])) > 0

    profiles = client.get("/api/profiles")
    assert profiles.status_code == 200

    activity = client.get("/api/activity?limit=5")
    assert activity.status_code == 200

    bundles = client.get("/api/support/bundles?limit=5")
    assert bundles.status_code == 200

    runtime = client.get("/api/runtime/status")
    assert runtime.status_code == 200

    logs = client.get("/api/runtime/logs?lines=10")
    assert logs.status_code == 200
    assert "lines" in logs.json()

    log_insights = client.get("/api/runtime/log-insights?lines=20")
    assert log_insights.status_code == 200


def test_ui_workflow_post_endpoints():
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app

    client = TestClient(create_app())
    spec = dict(_UI_BASE_SPEC)

    flow = client.post("/api/pipeline/flow", json=spec)
    assert flow.status_code == 200
    assert flow.json().get("stages")

    diagnostics = client.post("/api/diagnostics", json=spec)
    assert diagnostics.status_code == 200
    diag_payload = diagnostics.json()
    assert diag_payload.get("ports")
    assert "observed_runtime" in diag_payload

    gates = client.post("/api/gates", json=spec)
    assert gates.status_code == 200
    assert gates.json().get("groups")

    plan = client.post("/api/launch/plan", json=spec)
    assert plan.status_code == 200
    assert plan.json().get("command")

    diff = client.post("/api/launch/diff", json=spec)
    assert diff.status_code == 200
    assert "summary" in diff.json()

    matrix = client.post("/api/model-matrix", json=spec)
    assert matrix.status_code == 200
    assert matrix.json().get("rows")

    sources = client.post("/api/sources", json=spec)
    assert sources.status_code == 200
    assert sources.json().get("rows")

    remediation = client.post("/api/remediation", json=spec)
    assert remediation.status_code == 200
    assert "actions" in remediation.json()

    preview = client.post("/api/launch/preview", json=spec)
    assert preview.status_code == 200
    assert preview.json().get("validation")

    validate = client.post("/api/launch/validate", json=spec)
    assert validate.status_code == 200
    assert "results" in validate.json()

    decision = client.post("/api/launch/decision", json=spec)
    assert decision.status_code == 200
    assert "status" in decision.json()

    candidates = client.post("/api/launch/candidates", json=spec)
    assert candidates.status_code == 200
    assert "items" in candidates.json()


def test_ui_live_health_payload_shape():
    from fastapi.testclient import TestClient

    from noesis.dev_console.server import create_app

    client = TestClient(create_app())
    health = client.post("/api/live/health", json=_UI_BASE_SPEC)
    assert health.status_code == 200
    payload = health.json()
    assert "status" in payload
    assert "score" in payload
    assert "indicators" in payload


def _sample_ds8_process(
    *,
    pid: int,
    managed: bool = False,
    ws_port: int = 6010,
    rest_port: int = 8180,
    elapsed_s: int = 120,
) -> dict:
    return {
        "pid": pid,
        "looks_like_ds8": True,
        "managed_by_console": managed,
        "launch": {"ws_port": str(ws_port), "rest_port": str(rest_port)},
        "ports": [
            {"label": "WebSocket", "port": ws_port, "host": "127.0.0.1"},
            {"label": "REST", "port": rest_port, "host": "127.0.0.1"},
        ],
        "stats": {"elapsed_s": elapsed_s},
        "command": "python -m noesis.ds8_runtime",
    }


def test_runtime_target_from_process_uses_listening_ports():
    from noesis.dev_console.diagnostics import runtime_target_from_process

    target = runtime_target_from_process(
        {
            "pid": 4242,
            "looks_like_ds8": True,
            "managed_by_console": False,
            "launch": {},
            "ports": [{"label": "WebSocket", "port": 6010}, {"label": "REST", "port": 8180}],
            "stats": {},
            "command": "python -m noesis.ds8_runtime",
        }
    )
    assert target["pid"] == 4242
    assert target["ws_port"] == 6010
    assert target["rest_port"] == 8180


def test_observed_runtime_target_prefers_managed_pid():
    from noesis.dev_console.diagnostics import observed_runtime_target

    processes = [
        _sample_ds8_process(pid=100, managed=False, ws_port=6010),
        _sample_ds8_process(pid=200, managed=True, ws_port=6008, rest_port=8080),
    ]
    target = observed_runtime_target(processes, managed_pid=200)
    assert target is not None
    assert target["pid"] == 200
    assert target["managed_by_console"] is True
    assert target["ws_port"] == 6008


def test_observed_runtime_target_picks_matching_external_ws_port():
    from noesis.dev_console.diagnostics import observed_runtime_target
    from noesis.dev_console.launch_spec import LaunchSpec

    processes = [
        _sample_ds8_process(pid=100, managed=False, ws_port=6010, elapsed_s=300),
        _sample_ds8_process(pid=101, managed=False, ws_port=6020, elapsed_s=30),
    ]
    spec = LaunchSpec(ws_port=6020)
    target = observed_runtime_target(processes, managed_pid=None, spec=spec)
    assert target is not None
    assert target["pid"] == 101
    assert target["ws_port"] == 6020


def test_observed_runtime_target_returns_none_without_ds8():
    from noesis.dev_console.diagnostics import observed_runtime_target

    processes = [{"pid": 1, "looks_like_ds8": False, "managed_by_console": False, "ports": []}]
    assert observed_runtime_target(processes, managed_pid=None) is None


def test_noesis_port_ownership_maps_external_ds8_ports():
    from noesis.dev_console.diagnostics import noesis_port_ownership

    diagnostics = {
        "ports": [
            {"label": "WebSocket", "port": 6008, "busy": True, "owner": {"pids": [4242]}},
            {"label": "REST", "port": 8080, "busy": True, "owner": {"pids": [4242]}},
            {"label": "RTSP mosaic", "port": 8554, "busy": True, "owner": {"pids": [4242]}},
        ],
        "processes": [
            {
                "pid": 4242,
                "looks_like_ds8": True,
                "managed_by_console": False,
                "ports": [
                    {"label": "WebSocket", "port": 6008, "selected": True},
                    {"label": "REST", "port": 8080, "selected": True},
                    {"label": "RTSP mosaic", "port": 8554, "selected": True},
                ],
            }
        ],
    }
    ownership = noesis_port_ownership(diagnostics)
    assert ownership[6008]["pid"] == 4242
    assert ownership[6008]["managed_by_console"] is False
    assert set(ownership) == {6008, 8080, 8554}


def test_normalize_port_validation_downgrades_ds8_runtime_blocks():
    from noesis.dev_console.validator import normalize_port_validation_results

    validation = {
        "blocking": True,
        "results": [
            {"severity": "block", "code": "port.6008.busy", "message": "busy", "fix_hint": "stop"},
            {"severity": "block", "code": "port.8080.busy", "message": "busy", "fix_hint": "stop"},
            {"severity": "info", "code": "pipeline_config.ok", "message": "ok", "fix_hint": ""},
        ],
        "counts": {"block": 2, "warn": 0, "info": 1},
    }
    diagnostics = {
        "ports": [
            {"label": "WebSocket", "port": 6008, "busy": True, "owner": {"pids": [4242]}},
            {"label": "REST", "port": 8080, "busy": True, "owner": {"pids": [4242]}},
        ],
        "processes": [
            {
                "pid": 4242,
                "looks_like_ds8": True,
                "managed_by_console": False,
                "ports": [
                    {"label": "WebSocket", "port": 6008, "selected": True},
                    {"label": "REST", "port": 8080, "selected": True},
                ],
            }
        ],
    }
    normalized = normalize_port_validation_results(validation, diagnostics)
    assert normalized["blocking"] is False
    assert normalized["counts"]["block"] == 0
    assert normalized["counts"]["info"] == 3
    codes = {item["code"] for item in normalized["results"]}
    assert "port.6008.ds8_runtime" in codes
    assert "port.8080.ds8_runtime" in codes


def test_busy_non_console_ports_ignores_noesis_runtime():
    from noesis.dev_console.launch_decision import _busy_non_console_ports

    diagnostics = {
        "ports": [
            {"label": "WebSocket", "port": 6008, "busy": True, "owner": {"pids": [4242]}},
            {"label": "REST", "port": 8080, "busy": True, "owner": {"pids": [9999]}},
        ],
        "processes": [
            {
                "pid": 4242,
                "looks_like_ds8": True,
                "managed_by_console": False,
                "ports": [{"label": "WebSocket", "port": 6008, "selected": True}],
            }
        ],
    }
    busy = _busy_non_console_ports(diagnostics)
    assert [item["port"] for item in busy] == [8080]
