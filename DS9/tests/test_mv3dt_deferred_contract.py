from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DS9_ROOT = REPO_ROOT / "DS9"


def _load_launcher():
    path = DS9_ROOT / "noesis" / "ds9_runtime.py"
    spec = importlib.util.spec_from_file_location("ds9_mv3dt_deferred_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launcher_keeps_mv3dt_distinct_and_fails_before_preflight() -> None:
    launcher = _load_launcher()
    with mock.patch.object(
        sys,
        "argv",
        ["ds9_runtime.py", "--v3dt", "--tracking-mode", "mv3dt"],
    ):
        assert launcher._requested_tracking_mode() == "mv3dt"

    with (
        mock.patch.object(sys, "argv", ["ds9_runtime.py", "--tracking-mode", "mv3dt"]),
        mock.patch.dict(os.environ, {}, clear=False),
    ):
        os.environ.pop("NOESIS_DS9_PIPELINE_CONFIG", None)
        os.environ.pop("NOESIS_CAMERAS_CONFIG", None)
        assert launcher._requested_tracking_mode() == "mv3dt"
        pipeline, cameras, pipeline_explicit, cameras_explicit = (
            launcher._selected_launch_paths()
        )
        assert pipeline == (DS9_ROOT / "config" / "infer_mv3dt.yaml").resolve()
        assert cameras == (DS9_ROOT / "config" / "cameras_v3dt.yaml").resolve()
        assert pipeline_explicit is False
        assert cameras_explicit is False

        stderr = io.StringIO()
        with (
            mock.patch.object(launcher, "_ensure_runtime_sys_path"),
            mock.patch.object(launcher, "_set_ds9_environment") as set_environment,
            mock.patch.object(launcher, "_run_preflight") as preflight,
            contextlib.redirect_stderr(stderr),
        ):
            assert launcher.main() == 78
        set_environment.assert_not_called()
        preflight.assert_not_called()
        assert "MV3DT activation is deferred" in stderr.getvalue()


def test_core_and_hooks_preserve_mv3dt_mode_and_core_refuses_activation() -> None:
    code = r'''
import sys
from types import SimpleNamespace
from unittest import mock

with (
    mock.patch("noesis.native_artifact_provenance.attest_ds9_native_artifacts"),
    mock.patch("noesis.runtime_paths.require_ds9_native_extension_origins"),
):
    from noesis import ds9_runtime_core as runtime
    from noesis.pipelines import hooks

assert runtime._normalize_tracking_mode("mv3dt") == "mv3dt"
assert hooks._AnalyticsTelemetryProcessor._normalize_tracking_mode("mv3dt") == "mv3dt"
try:
    hooks._AnalyticsTelemetryProcessor._normalize_tracking_mode("mv3dtt")
except ValueError:
    pass
else:
    raise AssertionError("hooks accepted an unknown tracking mode")

try:
    hooks._AnalyticsTelemetryProcessor(
        pipeline=SimpleNamespace(
            config={
                "models": {},
                "tracking_mode": "mv3dt",
                "v3dt": {
                    "profile": "mv3dt",
                    "activation_state": "deferred",
                    "world_frame": "backend_world_m",
                    "caminfo_world_axes": "xzy",
                },
            }
        ),
        tracking_pub=object(),
        camera_labels={0: "living-room"},
        sensor_id_map={},
        tracking_mode="mv3dt",
    )
except ValueError as exc:
    assert "MV3DT activation is deferred" in str(exc)
else:
    raise AssertionError("hooks allowed direct MV3DT activation")

sys.argv = ["ds9_runtime_core.py", "--tracking-mode", "mv3dt"]
with mock.patch.object(
    runtime,
    "_select_ws_port",
    side_effect=AssertionError("MV3DT guard ran after endpoint preflight"),
):
    assert runtime._run_main(SimpleNamespace()) == 78
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30.0,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mv3dt_config_is_deferred_and_has_only_kitchen_family_peer_edge() -> None:
    pipeline_path = DS9_ROOT / "config" / "infer_mv3dt.yaml"
    tracker_path = DS9_ROOT / "config" / "v3dt" / "nvtracker_mv3dt.yaml"
    pub_sub_path = DS9_ROOT / "config" / "v3dt" / "pub_sub_info_config_0.yml"

    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    tracker = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
    pub_sub = yaml.safe_load(pub_sub_path.read_text(encoding="utf-8"))

    assert pipeline["v3dt"]["profile"] == "mv3dt"
    assert pipeline["v3dt"]["activation_state"] == "deferred"
    assert pipeline["streammux"]["sync-inputs"] == 1
    assert pipeline["tracker"]["config-file"].endswith("nvtracker_mv3dt.yaml")

    associator = tracker["MultiViewAssociator"]
    communicator = tracker["Communicator"]
    assert "enable" not in associator
    assert associator["multiViewAssociatorType"] == 1
    assert communicator["communicatorType"] == 2

    living, kitchen, family = pub_sub["pubBrokerTopicStr"]
    assert pub_sub["subPeerBrokerTopicStrs"] == [[], [family], [kitchen]]
    assert living not in {
        topic
        for subscriptions in pub_sub["subPeerBrokerTopicStrs"]
        for topic in subscriptions
    }


def test_mv3dt_asset_validator_accepts_only_the_explicit_mv3dt_profile() -> None:
    code = r'''
from pathlib import Path
from noesis.v3dt_assets import V3DTAssetError, validate_v3dt_assets

bundle = validate_v3dt_assets(
    Path("DS9/config/infer_mv3dt.yaml"),
    cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
    require_engines=False,
    require_sources=False,
    expected_profile="mv3dt",
)
assert bundle.profile == "mv3dt"
assert bundle.pub_sub_config.name == "pub_sub_info_config_0.yml"
assert bundle.mqtt_config_template.name == "config_mqtt.template.txt"

try:
    validate_v3dt_assets(
        Path("DS9/config/infer_mv3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        require_engines=False,
        require_sources=False,
        expected_profile="sv3dt",
    )
except V3DTAssetError as exc:
    assert "expected sv3dt" in str(exc)
else:
    raise AssertionError("MV3DT assets were accepted as the SV3DT profile")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(DS9_ROOT), str(REPO_ROOT), env.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30.0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
