from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import ds8_runtime_30s_gate as gate


def _prepared_state(tmp_path: Path) -> gate.CanaryStatePaths:
    repo = tmp_path / "repo"
    config = repo / "config"
    config.mkdir(parents=True)
    (config / "nvdsanalytics.yaml").write_text(
        "analytics:\n  stages:\n    exclude: {}\n",
        encoding="utf-8",
    )
    (config / "camera_calibration.json").write_text("{}\n", encoding="utf-8")
    (config / "ply_alignment.json").write_text("{}\n", encoding="utf-8")
    return gate._prepare_canary_state(tmp_path / "state", repo_root=repo)


def _secret_paths(tmp_path: Path) -> gate.RuntimeSecretPaths:
    operator = tmp_path / "operator"
    return gate.RuntimeSecretPaths(
        camera_sources=operator / "camera_sources.json",
        mapanything_key=operator / "mapanything_rpc.key",
        internal_auth_token=operator / "gateway-token",
    )


def test_private_atomic_report_persists_exact_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / "evidence"
    report_dir.mkdir(mode=0o700)
    report = report_dir / "canary.json"
    payload = {"ok": True, "scope": "test", "process_exit": 0}

    gate._emit_report(payload, report)

    assert json.loads(report.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(report.stat().st_mode) == 0o600
    assert report.stat().st_nlink == 1
    assert json.loads(capsys.readouterr().out) == payload


def test_private_atomic_report_rejects_symlink_destination(tmp_path: Path) -> None:
    report_dir = tmp_path / "evidence"
    report_dir.mkdir(mode=0o700)
    target = report_dir / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o600)
    report = report_dir / "canary.json"
    report.symlink_to(target)

    with pytest.raises(RuntimeError, match="symlink"):
        gate._emit_report({"ok": False}, report)


def _health_payload(
    *, tracking_sequence: int = 10, world_sequence: int = 20
) -> dict[str, object]:
    return {
        "contract": "noesis.capability.health",
        "contract_version": 1,
        "instance_id": "instance-1",
        "run_id": "run-1",
        "generated_at_us": 1_000_000,
        "capabilities": [
            {
                "capability": "tracking_observations",
                "status": "healthy",
                "checked_at_us": 1_000_000,
                "last_success_at_us": 999_999,
                "evidence": {"sequence": tracking_sequence},
                "blockers": [],
            },
            {
                "capability": "global_world",
                "status": "healthy",
                "checked_at_us": 1_000_000,
                "last_success_at_us": 999_999,
                "evidence": {"sequence": world_sequence},
                "blockers": [],
            },
        ],
    }


def _cli_values(command: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if (
            item.startswith("--")
            and index + 1 < len(command)
            and not command[index + 1].startswith("--")
        ):
            values[item] = command[index + 1]
            index += 2
            continue
        index += 1
    return values


def test_runtime_command_is_explicit_canonical_yolo26m_baseline(tmp_path: Path) -> None:
    storage_base = tmp_path / "state" / "depth"
    command = gate._build_runtime_command(
        repo_root=tmp_path,
        ws_host="127.0.0.1",
        ws_port=6008,
        rest_host="127.0.0.1",
        rest_port=8080,
        storage_base=storage_base,
        python_bin="/usr/bin/python-test",
    )

    assert command[:4] == [
        "/usr/bin/python-test",
        "-X",
        "faulthandler",
        str(tmp_path / "noesis" / "ds8_runtime.py"),
    ]
    values = _cli_values(command)
    assert values == {
        "--pipeline-config": str(tmp_path / "config" / "infer.yaml"),
        "--cameras-config": str(tmp_path / "config" / "cameras.yaml"),
        "--pgie-profile": "yolo26",
        "--size": "m",
        "--tracking-mode": "baseline",
        "--ws-host": "127.0.0.1",
        "--ws-port": "6008",
        "--rest-host": "127.0.0.1",
        "--rest-port": "8080",
        "--storage-base": str(storage_base),
        "--log-level": "INFO",
    }
    assert "--enable-rest" in command
    assert "--disable-rest" not in command


def test_runtime_command_is_explicit_fail_closed_v3dt_contract(
    tmp_path: Path,
) -> None:
    storage_base = tmp_path / "state" / "depth"
    command = gate._build_runtime_command(
        repo_root=tmp_path,
        ws_host="127.0.0.1",
        ws_port=6008,
        rest_host="127.0.0.1",
        rest_port=8080,
        storage_base=storage_base,
        python_bin="/usr/bin/python-test",
        runtime_contract=gate.V3DT_RUNTIME_CONTRACT,
    )

    assert command[:4] == [
        "/usr/bin/python-test",
        "-X",
        "faulthandler",
        str(tmp_path / "noesis" / "ds8_runtime_v3dt_reimpl.py"),
    ]
    values = _cli_values(command)
    assert values == {
        "--pipeline-config": str(
            tmp_path / "config" / "infer_v3dt_reimpl_fast1056_mp4.yaml"
        ),
        "--cameras-config": str(
            tmp_path / "config" / "cameras_v3dt_baseline.yaml"
        ),
        "--pgie-profile": "yolo26_seg",
        "--size": "s",
        "--tracking-mode": "v3dt",
        "--ws-host": "127.0.0.1",
        "--ws-port": "6008",
        "--rest-host": "127.0.0.1",
        "--rest-port": "8080",
        "--storage-base": str(storage_base),
        "--log-level": "INFO",
    }
    assert "--enable-rest" in command
    assert "--disable-rest" not in command


def test_runtime_environment_removes_degraded_and_secret_inputs(tmp_path: Path) -> None:
    poisoned = {key: "poison" for key in gate.SANITIZED_ENV_KEYS}
    poisoned.update(
        {
            "NOESIS_INTERNAL_AUTH_MODE": "disabled",
            "NOESIS_INTERNAL_AUTH_TOKEN_FILE": "/wrong/token",
            "NOESIS_PGIE_PROFILE": "yolo11",
            "NOESIS_TRACKING_MODE": "v3dt",
            "NOESIS_MOSAIC_RTSP_ENABLED": "0",
            "NOESIS_MOSAIC_WEBRTC_ENABLED": "0",
            "NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS": "99",
            "NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS": "0",
            "NOESIS_REID_ENABLED": "0",
            "NOESIS_SHUTDOWN_GRACE_SECONDS": "5",
            "SAFE_VALUE": "preserved",
        }
    )
    state = _prepared_state(tmp_path)
    secret_paths = _secret_paths(tmp_path)

    child = gate._build_runtime_env(
        poisoned,
        state=state,
        secret_paths=secret_paths,
    )

    overridden = {
        "NOESIS_CAMERA_SECRETS_FILE",
        "NOESIS_MAPANYTHING_API_KEY_FILE",
        "NOESIS_ANALYTICS_CONFIG",
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG",
        "NOESIS_BUILD_DIR",
        "NOESIS_CALIBRATION_AUDIT_DIR",
        "NOESIS_CAMERA_CALIBRATION_FILE",
        "NOESIS_IDENTITY_V2_STORE",
        "NOESIS_REID_GALLERY_FILE",
        "NOESIS_REID_ALIAS_FILE",
        "NOESIS_REID_SID_POOL_FILE",
        "NOESIS_PLY_ALIGNMENT_FILE",
        "NOESIS_SCENE_STORE_PATH",
        "NOESIS_VIRTUAL_TWIN_ROOT",
        "NOESIS_WORLD_JOURNAL_PATH",
        "NOESIS_V3DT_DIAG_DIR",
        "NOESIS_MOSAIC_H264_SHM",
        "GST_DEBUG_DUMP_DOT_DIR",
        "GST_REGISTRY_1_0",
    }
    assert all(
        key not in child
        for key in gate.SANITIZED_ENV_KEYS
        if key not in overridden
    )
    assert "SAFE_VALUE" not in child
    assert child["NOESIS_INTERNAL_AUTH_MODE"] == "required"
    assert child["NOESIS_INTERNAL_AUTH_TOKEN_FILE"] == str(
        secret_paths.internal_auth_token
    )
    assert child["NOESIS_PGIE_PROFILE"] == "yolo26"
    assert child["NOESIS_TRACKING_MODE"] == "baseline"
    assert child["NOESIS_MOSAIC_RTSP_ENABLED"] == "0"
    assert child["NOESIS_MOSAIC_WEBRTC_ENABLED"] == "1"
    assert child["NOESIS_MOSAIC_WEBRTC_MAX_CLIENTS"] == "1"
    assert child["NOESIS_MOSAIC_WEBRTC_INITIAL_CLIENTS"] == "1"
    assert child["NOESIS_MOSAIC_H264_SHM"] == str(state.temporary / "mosaic-h264.sock")
    assert child["NOESIS_REID_ENABLED"] == "1"
    assert child["NOESIS_SHUTDOWN_GRACE_SECONDS"] == "75"


def test_decoded_media_probe_requires_smoke_test_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "gateway-token"
    token_file.write_text("secret", encoding="utf-8")
    captured: list[list[str]] = []

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "rtp_packets": 50,
                    "decoded_frames": 5,
                    "answer_video_direction": "sendonly",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", _run)

    payload = gate._probe_webrtc_decoded_media(
        ws_host="127.0.0.1",
        ws_port=6008,
        auth_token_file=token_file,
    )

    assert payload["decoded_frames"] == 5
    assert captured
    assert "--min-decoded" in captured[0]
    assert str(token_file) in captured[0]


def test_v3dt_runtime_environment_forces_canonical_meta_and_tracker(
    tmp_path: Path,
) -> None:
    poisoned = {key: "poison" for key in gate.SANITIZED_ENV_KEYS}
    poisoned.update(
        {
            "NOESIS_INTERNAL_AUTH_MODE": "disabled",
            "NOESIS_PGIE_PROFILE": "yolo11_seg",
            "NOESIS_TRACKING_MODE": "baseline",
            "NOESIS_V3DT_AUTOGEN_CAMINFO": "1",
            "NOESIS_V3DT_META_EXTRACT": "0",
            "NOESIS_V3DT_USE_PATCHED_NVTRACKER": "0",
        }
    )

    child = gate._build_runtime_env(
        poisoned,
        state=_prepared_state(tmp_path),
        secret_paths=_secret_paths(tmp_path),
        runtime_contract=gate.V3DT_RUNTIME_CONTRACT,
    )

    assert child["NOESIS_INTERNAL_AUTH_MODE"] == "required"
    assert child["NOESIS_PGIE_PROFILE"] == "yolo26_seg"
    assert child["NOESIS_TRACKING_MODE"] == "v3dt"
    assert child["NOESIS_V3DT_AUTOGEN_CAMINFO"] == "0"
    assert child["NOESIS_V3DT_META_EXTRACT"] == "1"
    assert child["NOESIS_V3DT_USE_PATCHED_NVTRACKER"] == "1"


def test_canary_state_is_private_external_and_reusable_without_reset(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    config = repo / "config"
    config.mkdir(parents=True)
    source_payloads = {
        "nvdsanalytics.yaml": b"analytics:\n  stages:\n    exclude: {}\n",
        "camera_calibration.json": b'{"camera":"source"}\n',
        "ply_alignment.json": b'{"alignment":"source"}\n',
    }
    for name, payload in source_payloads.items():
        (config / name).write_bytes(payload)
    source_before = {
        name: ((config / name).read_bytes(), (config / name).stat().st_mtime_ns)
        for name in source_payloads
    }

    state = gate._prepare_canary_state(tmp_path / "canary-state", repo_root=repo)
    state.identity_store.write_bytes(b"persistent-identity-state")
    state.identity_store.chmod(0o600)
    identity_inode = state.identity_store.stat().st_ino
    analytics_inode = state.analytics_config.stat().st_ino
    repeated = gate._prepare_canary_state(state.root, repo_root=repo)

    assert repeated == state
    assert repeated.identity_store.read_bytes() == b"persistent-identity-state"
    assert repeated.identity_store.stat().st_ino == identity_inode
    assert repeated.analytics_config.stat().st_ino == analytics_inode
    assert repeated.analytics_config.read_bytes() == source_payloads["nvdsanalytics.yaml"]
    assert repeated.camera_calibration.read_bytes() == source_payloads[
        "camera_calibration.json"
    ]
    assert repeated.ply_alignment.read_bytes() == source_payloads[
        "ply_alignment.json"
    ]
    assert (state.root / gate.CANARY_STATE_MARKER_NAME).read_bytes() == (
        gate.CANARY_STATE_MARKER
    )
    for directory in (path for path in state.root.rglob("*") if path.is_dir()):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for name, (payload, modified_ns) in source_before.items():
        assert (config / name).read_bytes() == payload
        assert (config / name).stat().st_mtime_ns == modified_ns


def test_canary_state_rejects_nonexternal_or_unowned_cohorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    operator_home = tmp_path / "operator-home"
    operator_home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(operator_home))

    with pytest.raises(RuntimeError, match="absolute"):
        gate._require_external_state_root(Path("relative-state"), repo_root=repo)
    with pytest.raises(RuntimeError, match="checkout"):
        gate._require_external_state_root(repo / "state", repo_root=repo)
    with pytest.raises(RuntimeError, match="checkout"):
        gate._require_external_state_root(tmp_path, repo_root=repo)
    with pytest.raises(RuntimeError, match="operator home"):
        gate._require_external_state_root(
            operator_home / "canary-state",
            repo_root=repo,
        )

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="mode must be 0700"):
        gate._require_external_state_root(unsafe, repo_root=repo)

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        gate._require_external_state_root(linked, repo_root=repo)

    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o700)
    (sealed / "SEALED.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed checkpoint"):
        gate._require_external_state_root(sealed / "state", repo_root=repo)

    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    (foreign / "unrelated").write_text("data\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected entries"):
        gate._require_external_state_root(foreign, repo_root=repo)


def test_runtime_environment_preserves_only_secret_file_authorities(
    tmp_path: Path,
) -> None:
    state = _prepared_state(tmp_path)
    operator = tmp_path / "operator"
    secret_dir = operator / "secrets"
    secret_dir.mkdir(parents=True, mode=0o700)
    camera_file = secret_dir / "camera_sources.json"
    mapanything_file = secret_dir / "mapanything_rpc.key"
    token_file = secret_dir / "gateway-token"
    secret_values = (
        "rtsp://user:camera-secret@example.invalid/live",
        "mapanything-secret-value",
        "internal-token-secret-value",
    )
    for path, value in zip(
        (camera_file, mapanything_file, token_file),
        secret_values,
        strict=True,
    ):
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    base_env = {
        "HOME": str(operator),
        "PATH": "/runtime/bin",
        "PYTHONPATH": "/runtime/python",
        "PYTHONUSERBASE": str(operator / ".local"),
        "NOESIS_CAMERA_SECRETS_FILE": str(camera_file),
        "NOESIS_MAPANYTHING_API_KEY_FILE": str(mapanything_file),
        "NOESIS_WORLD_JOURNAL_PATH": str(operator / "protected-world.sqlite3"),
        "SSH_AUTH_SOCK": str(operator / "ssh-agent.sock"),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/private-bus",
        "AWS_SECRET_ACCESS_KEY": "cloud-secret-value",
    }
    secret_paths = gate._resolve_runtime_secret_paths(
        base_env,
        token_file=token_file,
    )
    child = gate._build_runtime_env(
        base_env,
        state=state,
        secret_paths=secret_paths,
    )
    command = gate._build_runtime_command(
        repo_root=tmp_path / "repo",
        ws_host="127.0.0.1",
        ws_port=6008,
        rest_host="127.0.0.1",
        rest_port=8080,
        storage_base=state.depth,
    )

    assert child["HOME"] == str(state.home)
    assert child["PYTHONUSERBASE"] == str(state.python_user_base)
    assert child["NOESIS_CAMERA_SECRETS_FILE"] == str(camera_file)
    assert child["NOESIS_MAPANYTHING_API_KEY_FILE"] == str(mapanything_file)
    assert child["NOESIS_INTERNAL_AUTH_TOKEN_FILE"] == str(token_file)
    assert child["NOESIS_WORLD_JOURNAL_PATH"] == str(state.world_journal)
    assert child["NOESIS_CAMERA_CALIBRATION_FILE"] == str(
        state.camera_calibration
    )
    assert child["NOESIS_PLY_ALIGNMENT_FILE"] == str(state.ply_alignment)
    assert child["GST_REGISTRY_1_0"] == str(state.gst_registry)
    assert child["XDG_RUNTIME_DIR"] == str(state.xdg_runtime)
    assert child["XDG_DATA_HOME"] == str(state.xdg_data)
    assert child["PATH"] == "/runtime/bin"
    assert child["PYTHONPATH"] != "/runtime/python"
    assert str(gate.REPO_ROOT) not in child["PYTHONPATH"].split(os.pathsep)
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONSAFEPATH"] == "1"
    for forbidden in (
        "SSH_AUTH_SOCK",
        "DBUS_SESSION_BUS_ADDRESS",
        "AWS_SECRET_ACCESS_KEY",
        "NOESIS_IDENTITY_V2_EVIDENCE_PATH",
    ):
        assert forbidden not in child
    serialized_authority = json.dumps(child, sort_keys=True) + json.dumps(command)
    for value in (*secret_values, "cloud-secret-value"):
        assert value not in serialized_authority


def test_isolated_runtime_environment_can_import_installed_dependencies(
    tmp_path: Path,
) -> None:
    state = _prepared_state(tmp_path)
    child = gate._build_runtime_env(
        {"HOME": str(Path.home()), "PATH": os.environ.get("PATH", os.defpath)},
        state=state,
        secret_paths=_secret_paths(tmp_path),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fastapi, numpy, pydantic, torch, uvicorn, websockets, "
                "yaml, zarr"
            ),
        ],
        cwd=state.temporary,
        env=child,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(state.home) not in child["PYTHONPATH"].split(os.pathsep)


def test_runtime_pythonpath_excludes_checkout_and_expands_active_pth_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    operator_home = tmp_path / "operator"
    expanded_dependency = (
        operator_home / ".local/lib/python3.12/site-packages/example.egg"
    )
    expanded_dependency.mkdir(parents=True)
    live_checkout = tmp_path / "live-checkout"
    live_checkout.mkdir()
    monkeypatch.setattr(
        gate.sys,
        "path",
        [
            "",
            "relative-shadow",
            str(checkpoint),
            str(checkpoint / "noesis"),
            str(live_checkout),
            str(expanded_dependency),
            "/usr/lib/python3.12",
        ],
    )

    rendered = gate._trusted_runtime_pythonpath(
        repo_root=checkpoint,
        operator_home=operator_home,
    ).split(os.pathsep)

    assert str(expanded_dependency) in rendered
    assert "/usr/lib/python3.12" in rendered
    assert str(checkpoint) not in rendered
    assert str(checkpoint / "noesis") not in rendered
    assert str(live_checkout) not in rendered
    assert "relative-shadow" not in rendered


def test_missing_auth_token_is_load_only_and_creates_no_path(tmp_path: Path) -> None:
    token_file = tmp_path / "missing-parent" / "gateway-token"

    with pytest.raises(RuntimeError, match="missing"):
        gate._load_auth_token(token_file)

    assert not token_file.exists()
    assert not token_file.parent.exists()


def test_evidence_paths_are_fresh_private_and_state_bound(tmp_path: Path) -> None:
    state = _prepared_state(tmp_path)
    log_path = state.evidence / "baseline-unique.log"
    resolved = gate._resolve_evidence_path(
        log_path,
        state=state,
        default_prefix="unused",
        suffix=".log",
        label="DS8 canary log path",
    )
    with gate._open_private_log(resolved) as handle:
        handle.write("evidence\n")
    assert stat.S_IMODE(resolved.stat().st_mode) == 0o600

    with pytest.raises(RuntimeError, match="caller-unique"):
        gate._resolve_evidence_path(
            log_path,
            state=state,
            default_prefix="unused",
            suffix=".log",
            label="DS8 canary log path",
        )
    with pytest.raises(RuntimeError, match="evidence directory"):
        gate._resolve_evidence_path(
            tmp_path / "outside.log",
            state=state,
            default_prefix="unused",
            suffix=".log",
            label="DS8 canary log path",
        )


def test_runtime_sources_honor_external_calibration_and_v3dt_build_paths() -> None:
    for relative in (
        "noesis/ds8_runtime.py",
        "noesis/ds8_runtime_v3dt_reimpl.py",
        "DS9/noesis/ds9_runtime_core.py",
    ):
        source = (gate.REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "NOESIS_CAMERA_CALIBRATION_FILE" in source
        assert "NOESIS_PLY_ALIGNMENT_FILE" in source

    v3dt = (gate.REPO_ROOT / "noesis/ds8_runtime_v3dt_reimpl.py").read_text(
        encoding="utf-8"
    )
    assert "out_dir = _build_dir()" in v3dt
    assert 'os.environ.get("GST_REGISTRY_1_0", _build_dir()' in v3dt


def test_checked_in_runtime_contracts_are_live_secret_backed() -> None:
    gate._validate_runtime_contract(
        gate.BASELINE_RUNTIME_CONTRACT,
        repo_root=gate.REPO_ROOT,
    )
    gate._validate_runtime_contract(
        gate.V3DT_RUNTIME_CONTRACT,
        repo_root=gate.REPO_ROOT,
    )


def test_baseline_and_v3dt_canaries_pin_one_swin_identity_profile() -> None:
    baseline = gate.BASELINE_RUNTIME_CONTRACT
    v3dt = gate.V3DT_RUNTIME_CONTRACT

    assert (
        baseline.expected_reid_config,
        baseline.expected_reid_engine,
        baseline.expected_reid_layer,
        baseline.expected_reid_dimension,
    ) == (
        v3dt.expected_reid_config,
        v3dt.expected_reid_engine,
        v3dt.expected_reid_layer,
        v3dt.expected_reid_dimension,
    )
    assert baseline.expected_reid_config == (
        "pipelines/config_infer_secondary_reid_swin.ini"
    )
    assert baseline.expected_reid_layer == "fc_pred"
    assert baseline.expected_reid_dimension == 256


def test_runtime_contract_rejects_inline_source_uri(tmp_path: Path) -> None:
    for relative in (
        gate.V3DT_RUNTIME_CONTRACT.entrypoint,
        gate.V3DT_RUNTIME_CONTRACT.cameras_config,
        Path(gate.V3DT_RUNTIME_CONTRACT.expected_tracker_config or ""),
        *gate.V3DT_RUNTIME_CONTRACT.required_regular_artifacts,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    pipeline_path = tmp_path / gate.V3DT_RUNTIME_CONTRACT.pipeline_config
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_path.write_text(
        """
sources:
  - element: nvurisrcbin
    uri_secret: living-room
    uri: file:///tmp/not-live.mp4
streammux:
  live-source: 1
tracker:
  config-file: config/v3dt/reimpl/nvtracker_sv3dt_yolo26s_fast.yml
mosaic_output:
  rtsp_enabled: true
  mosaic_webrtc_enabled: true
v3dt:
  tracking_mode: v3dt
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="uri_secret exclusively"):
        gate._validate_runtime_contract(
            gate.V3DT_RUNTIME_CONTRACT,
            repo_root=tmp_path,
        )


def test_v3dt_runtime_contract_rejects_missing_patched_tracker(
    tmp_path: Path,
) -> None:
    for relative in (
        gate.V3DT_RUNTIME_CONTRACT.entrypoint,
        gate.V3DT_RUNTIME_CONTRACT.pipeline_config,
        gate.V3DT_RUNTIME_CONTRACT.cameras_config,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="regular, non-symlink"):
        gate._validate_runtime_contract(
            gate.V3DT_RUNTIME_CONTRACT,
            repo_root=tmp_path,
        )


def test_v3dt_runtime_contract_rejects_symlinked_patched_tracker(
    tmp_path: Path,
) -> None:
    for relative in (
        gate.V3DT_RUNTIME_CONTRACT.entrypoint,
        gate.V3DT_RUNTIME_CONTRACT.pipeline_config,
        gate.V3DT_RUNTIME_CONTRACT.cameras_config,
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    target = tmp_path / "patched-tracker-target.so"
    target.write_text("fixture\n", encoding="utf-8")
    artifact = (
        tmp_path / gate.V3DT_RUNTIME_CONTRACT.required_regular_artifacts[0]
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.symlink_to(target)

    with pytest.raises(RuntimeError, match="regular, non-symlink"):
        gate._validate_runtime_contract(
            gate.V3DT_RUNTIME_CONTRACT,
            repo_root=tmp_path,
        )


def test_capability_validation_requires_healthy_advancing_producers() -> None:
    first = gate._validate_capability_payload(_health_payload())
    second = gate._validate_capability_payload(
        _health_payload(tracking_sequence=11, world_sequence=21)
    )

    gate._require_advancement(first, second)
    with pytest.raises(ValueError, match="did not advance"):
        gate._require_advancement(first, first)

    unhealthy = _health_payload()
    capabilities = unhealthy["capabilities"]
    assert isinstance(capabilities, list)
    assert isinstance(capabilities[0], dict)
    capabilities[0]["status"] = "degraded"
    with pytest.raises(ValueError, match="not healthy"):
        gate._validate_capability_payload(unhealthy)


def test_log_scan_requires_production_markers_and_rejects_errors(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text("\n".join(gate.REQUIRED_LOG_MARKERS) + "\n", encoding="utf-8")

    signatures, missing, _samples = gate._scan_log(log_path)
    assert signatures == []
    assert missing == []

    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "ERROR runtime failed\n",
        encoding="utf-8",
    )
    signatures, missing, samples = gate._scan_log(log_path)
    assert signatures == ["ERROR"]
    assert missing == []
    assert samples == ["ERROR runtime failed"]


def test_log_scan_uses_v3dt_specific_runtime_markers(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "\n".join(gate.V3DT_RUNTIME_CONTRACT.required_log_markers) + "\n",
        encoding="utf-8",
    )

    signatures, missing, _samples = gate._scan_log(
        log_path,
        required_markers=gate.V3DT_RUNTIME_CONTRACT.required_log_markers,
    )

    assert signatures == []
    assert missing == []


def test_log_scan_accepts_callback_before_receipt_log(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    markers = list(gate.REQUIRED_LOG_MARKERS)
    orderly = "Orderly pipeline EOS accepted:"
    eos = "EOS received on pipeline (reason=shutdown_requested)"
    orderly_index = markers.index(orderly)
    eos_index = markers.index(eos)
    markers[orderly_index], markers[eos_index] = eos, orderly
    log_path.write_text("\n".join(markers) + "\n", encoding="utf-8")

    signatures, missing, _samples = gate._scan_log(log_path)

    assert signatures == []
    assert missing == []


def test_log_scan_rejects_callback_before_request_initiation(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    markers = list(gate.REQUIRED_LOG_MARKERS)
    request = "Orderly pipeline EOS request initiated"
    accepted = "Orderly pipeline EOS accepted:"
    callback = "EOS received on pipeline (reason=shutdown_requested)"
    markers.remove(request)
    markers.remove(accepted)
    callback_index = markers.index(callback)
    markers.insert(callback_index + 1, request)
    markers.insert(callback_index + 2, accepted)
    log_path.write_text("\n".join(markers) + "\n", encoding="utf-8")

    signatures, missing, _samples = gate._scan_log(log_path)

    assert signatures == []
    assert missing == [
        "orderly_eos_requested_before_shutdown_eos_callback"
    ]


def test_graceful_sigterm_requires_exact_zero_exit() -> None:
    code = """
import signal
import sys
import time

def stop(_signum, _frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
while True:
    time.sleep(0.05)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"

    result = gate._stop_process(proc, timeout_s=2.0)

    assert result.returncode == 0
    assert result.forced_kill is False
    assert result.error == ""
    assert result.signal_sent is True


def test_ignored_sigterm_is_forced_and_cannot_pass() -> None:
    code = """
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready', flush=True)
while True:
    time.sleep(0.05)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    try:
        result = gate._stop_process(proc, timeout_s=0.1)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2.0)

    assert result.forced_kill is True
    assert result.returncode == -signal.SIGKILL
    assert "SIGKILL" in result.error
    assert not gate._gate_ok(
        readiness_ok=True,
        active_completed=True,
        active_probe_ok=True,
        running_before_shutdown=True,
        stop=result,
        signatures_found=(),
        markers_missing=(),
    )


def test_gate_predicate_rejects_nonzero_exit_even_without_log_errors() -> None:
    assert gate._gate_ok(
        readiness_ok=True,
        active_completed=True,
        active_probe_ok=True,
        running_before_shutdown=True,
        stop=gate.StopResult(
            returncode=0,
            elapsed_s=0.1,
            forced_kill=False,
            signal_sent=True,
        ),
        signatures_found=(),
        markers_missing=(),
    )
    assert not gate._gate_ok(
        readiness_ok=True,
        active_completed=True,
        active_probe_ok=True,
        running_before_shutdown=True,
        stop=gate.StopResult(
            returncode=2,
            elapsed_s=0.1,
            forced_kill=False,
            signal_sent=True,
        ),
        signatures_found=(),
        markers_missing=(),
    )
