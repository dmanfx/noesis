from __future__ import annotations

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "DS9" / "scripts" / "ds9_live_validation_runner.py"
SPEC = importlib.util.spec_from_file_location(
    "ds9_live_validation_runner_test", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

SEMANTIC_FIXTURE_PATH = REPO_ROOT / "DS9" / "tests" / "test_semantic_observation_gate.py"
SEMANTIC_SPEC = importlib.util.spec_from_file_location(
    "ds9_live_runner_semantic_fixture",
    SEMANTIC_FIXTURE_PATH,
)
assert SEMANTIC_SPEC is not None and SEMANTIC_SPEC.loader is not None
semantic_fixture = importlib.util.module_from_spec(SEMANTIC_SPEC)
sys.modules[SEMANTIC_SPEC.name] = semantic_fixture
SEMANTIC_SPEC.loader.exec_module(semantic_fixture)


def _clean_shutdown_log() -> str:
    return "\n".join(
        (
            "INFO Orderly pipeline EOS accepted: component=orderly_eos_control request_sequence=1",
            "INFO EOS received on pipeline (reason=shutdown_requested)",
            "INFO pyservicemaker wait() returned (pipeline stopped)",
            "INFO Shutdown complete",
            "",
        )
    )


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "lane": "baseline",
        "session_id": "live-runner-test",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "ownership_evidence_dir": None,
        "skip": [],
        "no_spawn": False,
        "pipeline_config": Path("DS9/config/infer.yaml"),
        "cameras_config": Path("config/cameras.yaml"),
        "ws": "ws://127.0.0.1:6008",
        "rest": "http://127.0.0.1:8080",
        "rtsp_url": "rtsp://127.0.0.1:8554/mosaic",
        "startup_timeout_s": 90.0,
        "shutdown_timeout_s": 90.0,
        "rtsp_duration_s": 20.0,
        "webrtc_duration_s": 6.0,
        "webrtc_pt": 103,
        "webrtc_min_rtp": 10,
        "webrtc_min_decoded": 1,
        "reid_duration_s": 35.0,
        "semantic_observation_duration_s": 45.0,
        "identity_evidence": Path("/private/identity-v2.jsonl"),
        "identity_require_cross_camera": False,
        "identity_require_open_set": False,
        "bev_duration_s": 45.0,
        "bridge_duration_s": 75.0,
        "floorplan_max_age_sec": 120.0,
        "floorplan_timeout_s": 75.0,
        "ma_camera": "family-room",
        "zero_copy_stats_duration_s": 60.0,
        "zero_copy_rest_duration_s": 90.0,
        "v3dt_bbox_duration_s": 20.0,
        "v3dt_bbox_attempts": 8,
        "v3dt_world_duration_s": 45.0,
        "wholebody_duration_s": 45.0,
        "output_dir": Path("DS9/build/live_validation/test"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _health_payload(
    *, tracking_sequence: int = 10, world_sequence: int = 20
) -> dict[str, object]:
    return {
        "contract": "noesis.capability.health",
        "contract_version": 1,
        "instance_id": "ds9-instance",
        "run_id": "ds9-run",
        "generated_at_us": 1_000_000,
        "capabilities": [
            {
                "capability": "tracking_observations",
                "status": "healthy",
                "checked_at_us": 999_999,
                "last_success_at_us": 999_999,
                "evidence": {"sequence": tracking_sequence},
            },
            {
                "capability": "global_world",
                "status": "healthy",
                "checked_at_us": 999_999,
                "last_success_at_us": 999_999,
                "evidence": {"sequence": world_sequence},
            },
        ],
    }


def test_semantic_identity_source_must_be_complete_private_file(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = private / "identity_v2.jsonl"
    source.write_bytes(b'{"event_id":"event-a"}\n')
    source.chmod(0o600)
    args = _args(identity_evidence=source)

    runner._validate_identity_evidence_source(args)

    source.write_bytes(b'{"event_id":"event-a"}')
    source.chmod(0o600)
    with pytest.raises(ValueError, match="empty or incomplete"):
        runner._validate_identity_evidence_source(args)

    source.unlink()
    target = private / "target.jsonl"
    target.write_bytes(b'{"event_id":"event-a"}\n')
    target.chmod(0o600)
    source.symlink_to(target)
    with pytest.raises(Exception, match="symlink"):
        runner._validate_identity_evidence_source(args)


def test_behavior_output_directory_is_created_owner_private(tmp_path: Path) -> None:
    output = tmp_path / "behavior"

    prepared = runner._prepare_private_output_dir(output)

    assert prepared == output.absolute()
    assert (prepared.stat().st_mode & 0o777) == 0o700


def test_behavior_output_directory_rejects_insecure_or_linked_parent(
    tmp_path: Path,
) -> None:
    insecure = tmp_path / "insecure"
    insecure.mkdir(mode=0o755)
    insecure.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only mode 0700"):
        runner._prepare_private_output_dir(insecure)
    assert (insecure.stat().st_mode & 0o777) == 0o755

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="regular directory"):
        runner._prepare_private_output_dir(linked)


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


def test_ds9_runtime_command_is_explicit_yolo26m_baseline() -> None:
    command = runner._build_runtime_command(_args())
    values = _cli_values(command)

    assert command[:2] == [sys.executable, "DS9/noesis/ds9_runtime.py"]
    assert values["--pgie-profile"] == "yolo26"
    assert values["--size"] == "m"
    assert values["--tracking-mode"] == "baseline"
    assert values["--ws-host"] == "127.0.0.1"
    assert values["--ws-port"] == "6008"
    assert values["--rest-host"] == "127.0.0.1"
    assert values["--rest-port"] == "8080"
    assert values["--log-level"] == "INFO"
    assert "--enable-rest" in command


def test_ds9_v3dt_command_and_gate_bundle_are_exact_global_world_v2() -> None:
    args = _args(
        lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        ownership_evidence_dir=Path("/private/session/launcher"),
    )
    command = runner._build_runtime_command(args)
    values = _cli_values(command)
    assert values["--pipeline-config"] == "DS9/config/infer_v3dt.yaml"
    assert values["--cameras-config"] == "DS9/config/cameras_v3dt.yaml"
    assert values["--pgie-profile"] == "yolo26_seg"
    assert values["--size"] == "s"
    assert values["--tracking-mode"] == "v3dt"
    assert "DS9/config/infer.yaml" not in command

    gates = runner._build_gates(args, auth_token_file=Path("/private/token"))
    by_name = {name: gate for name, gate, *_rest in gates}
    rows_by_name = {name: row for name, *row in gates}
    assert set(by_name) == {
        "rtsp",
        "webrtc",
        "reid",
        "semantic-observation",
        "v3dt-bbox3d",
        "v3dt-world",
    }
    assert "DS9/scripts/sv3dt_meta_smoke_test.py" in by_name["v3dt-bbox3d"]
    assert _cli_values(by_name["v3dt-bbox3d"])["--attempts"] == "8"
    assert rows_by_name["v3dt-bbox3d"][1] == 187.0
    assert "DS9/scripts/v3dt_world_contract_smoke_test.py" in by_name["v3dt-world"]
    assert "scripts/menon_bev_track_parity_smoke_test.py" not in " ".join(
        by_name["v3dt-world"]
    )
    assert _cli_values(by_name["v3dt-world"])["--out"].endswith(
        "/v3dt-world-contract.json"
    )
    assert _cli_values(by_name["v3dt-world"])["--source-out"].endswith(
        "/v3dt-world-contract-source.json"
    )
    world_values = _cli_values(by_name["v3dt-world"])
    assert world_values["--launcher-evidence-dir"] == (
        "/private/session/launcher"
    )
    assert world_values["--pipeline-config"] == "DS9/config/infer_v3dt.yaml"
    assert world_values["--cameras-config"] == "DS9/config/cameras_v3dt.yaml"
    for name in (
        "webrtc",
        "reid",
        "semantic-observation",
        "v3dt-bbox3d",
        "v3dt-world",
    ):
        assert _cli_values(by_name[name])["--auth-token-file"] == "/private/token"


def test_ds9_v3dt_lane_rejects_baseline_or_arbitrary_config_paths() -> None:
    args = _args(
        lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
    )
    runner._validate_runtime_contract_args(args)
    with pytest.raises(ValueError, match="hard-pinned"):
        runner._validate_runtime_contract_args(
            _args(
                lane="v3dt",
                pipeline_config=Path("DS9/config/infer.yaml"),
                cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
            )
        )
    with pytest.raises(ValueError, match="unsupported"):
        runner._validation_lane(_args(lane="v3dtt"))
    with pytest.raises(ValueError, match="attempts"):
        runner._validate_runtime_contract_args(
            _args(
                lane="v3dt",
                pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
                cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
                v3dt_bbox_attempts=0,
            )
        )


@pytest.mark.parametrize("gate", sorted(runner.V3DT_PROMOTION_REQUIRED_GATES))
def test_ownership_attached_v3dt_rejects_every_required_gate_skip(
    gate: str,
) -> None:
    with pytest.raises(ValueError, match="promotion-required"):
        runner._validate_runtime_contract_args(
            _args(
                lane="v3dt",
                pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
                cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
                ownership_evidence_dir=Path("/private/session/launcher"),
                no_spawn=True,
                skip=[gate],
            )
        )


def test_unattached_v3dt_gate_is_explicitly_non_promotable_without_launch_plan() -> None:
    args = _args(
        lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        output_dir=Path("DS9/build/live_validation/unattached"),
    )
    gates = runner._build_gates(args, auth_token_file=Path("/private/token"))
    world = next(command for name, command, *_rest in gates if name == "v3dt-world")
    values = _cli_values(world)
    assert values["--launcher-evidence-dir"] == (
        "DS9/build/live_validation/unattached"
    )
    assert values["--out"].startswith("DS9/build/live_validation/unattached/")


@pytest.mark.parametrize(
    ("lane_name", "size", "mode"),
    (("wholebody49-s", "s", "masks"), ("wholebody49-x", "x", "boxes")),
)
def test_wholebody49_runner_lanes_are_exact_and_use_occupied_scene_gate(
    lane_name: str, size: str, mode: str
) -> None:
    args = _args(lane=lane_name)
    command = runner._build_runtime_command(args)
    values = _cli_values(command)
    assert values["--pgie-profile"] == "wholebody49"
    assert values["--size"] == size
    assert values["--tracking-mode"] == "baseline"
    gates = runner._build_gates(args, auth_token_file=Path("/private/token"))
    by_name = {name: gate for name, gate, *_rest in gates}
    assert set(by_name) == {"rtsp", "webrtc", "reid", "wholebody-occupied"}
    occupied = _cli_values(by_name["wholebody-occupied"])
    media = _cli_values(by_name["webrtc"])
    assert by_name["webrtc"][:2] == [
        sys.executable,
        "DS9/scripts/wholebody49_media_decode_gate.py",
    ]
    assert media["--session-id"] == "live-runner-test"
    assert media["--runtime-lane"] == lane_name
    assert media["--runtime-instance-id"] == "runtime-instance-test"
    assert media["--runtime-run-id"] == "runtime-run-test"
    assert media["--out"].endswith("/wholebody49-media-decode.json")
    assert media["--source-out"].endswith(
        "/wholebody49-media-decode-source.json"
    )
    assert occupied["--mode"] == mode
    assert occupied["--session-id"] == "live-runner-test"
    assert occupied["--runtime-lane"] == lane_name
    assert occupied["--runtime-instance-id"] == "runtime-instance-test"
    assert occupied["--runtime-run-id"] == "runtime-run-test"
    assert occupied["--out"].endswith("/wholebody49-occupied-scene.json")
    assert occupied["--source-out"].endswith(
        "/wholebody49-occupied-scene-source.json"
    )
    assert occupied["--auth-token-file"] == "/private/token"
    assert by_name["wholebody-occupied"].count("--expected-source-id") == 3
    assert occupied["--expected-source-id"] == "2"


def test_ds9_runtime_environment_is_sanitized_and_protected(tmp_path: Path) -> None:
    poisoned = {key: "poison" for key in runner.SANITIZED_ENV_KEYS}
    poisoned.update(
        {
            "NOESIS_INTERNAL_AUTH_MODE": "disabled",
            "NOESIS_PGIE_PROFILE": "yolo11_seg",
            "NOESIS_TRACKING_MODE": "v3dt",
            "SAFE_VALUE": "preserved",
        }
    )
    token_file = tmp_path / "gateway-token"

    env = runner._runtime_env(
        ["NOESIS_POSE_FEATURE_DEBUG=0", "EXTRA_SAFE=1"],
        token_file=token_file,
        session_id="live-runner-test",
        identity_evidence_path=tmp_path / "identity_v2.jsonl",
        base_env=poisoned,
    )

    assert all(
        key not in env
        for key in runner.SANITIZED_ENV_KEYS
        if key
        not in {
            "NOESIS_IDENTITY_V2_MODE",
            "NOESIS_IDENTITY_V2_EVIDENCE_PATH",
            "NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID",
            "NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME",
        }
    )
    assert env["SAFE_VALUE"] == "preserved"
    assert env["EXTRA_SAFE"] == "1"
    assert env["NOESIS_POSE_FEATURE_DEBUG"] == "0"
    assert env["NOESIS_INTERNAL_AUTH_MODE"] == "required"
    assert env["NOESIS_INTERNAL_AUTH_TOKEN_FILE"] == str(token_file)
    assert env["NOESIS_PGIE_PROFILE"] == "yolo26"
    assert env["NOESIS_TRACKING_MODE"] == "baseline"
    assert env["NOESIS_MOSAIC_RTSP_ENABLED"] == "1"
    assert env["NOESIS_MOSAIC_WEBRTC_ENABLED"] == "1"
    assert env["NOESIS_REID_ENABLED"] == "1"
    assert env["NOESIS_IDENTITY_V2_MODE"] == "shadow"
    assert env["NOESIS_IDENTITY_V2_EVIDENCE_PATH"] == str(
        tmp_path / "identity_v2.jsonl"
    )
    assert env["NOESIS_IDENTITY_V2_EVIDENCE_SESSION_ID"] == "live-runner-test"
    assert env["NOESIS_IDENTITY_V2_EVIDENCE_RUNTIME"] == "ds9"
    assert env["NOESIS_SHUTDOWN_GRACE_SECONDS"] == "75"

    with pytest.raises(SystemExit, match="cannot override"):
        runner._runtime_env(
            ["NOESIS_SKIP_CUDA_PREFLIGHT=1"],
            token_file=token_file,
            session_id="live-runner-test",
            identity_evidence_path=tmp_path / "identity_v2.jsonl",
            base_env={},
        )


def test_ds9_readiness_requires_healthy_advancing_capabilities() -> None:
    first = runner._validate_capability_payload(_health_payload())
    second = runner._validate_capability_payload(
        _health_payload(tracking_sequence=11, world_sequence=21)
    )

    runner._require_advancement(first, second)
    with pytest.raises(ValueError, match="did not advance"):
        runner._require_advancement(first, first)

    unhealthy = _health_payload()
    capabilities = unhealthy["capabilities"]
    assert isinstance(capabilities, list)
    assert isinstance(capabilities[1], dict)
    capabilities[1]["status"] = "blocked"
    capabilities[1]["blockers"] = ["fixture blocker"]
    with pytest.raises(ValueError, match="not healthy"):
        runner._validate_capability_payload(unhealthy)

    invalid_contract = _health_payload()
    invalid_capabilities = invalid_contract["capabilities"]
    assert isinstance(invalid_capabilities, list)
    assert isinstance(invalid_capabilities[0], dict)
    invalid_capabilities[0].pop("checked_at_us")
    with pytest.raises(ValueError, match="checked_at_us"):
        runner._validate_capability_payload(invalid_contract)

    attached = _args()
    attached.supervisor_runtime_instance_id = "other-instance"
    attached.supervisor_runtime_run_id = "ds9-run"
    with pytest.raises(ValueError, match="differs from supervisor"):
        runner._bind_runtime_identity(attached, first)


def test_authenticated_capability_health_rejects_duplicate_json_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = json.dumps(_health_payload()).replace(
        '"instance_id": "ds9-instance"',
        '"instance_id": "ds9-instance", "instance_id": "ds9-instance"',
        1,
    ).encode()

    class Response:
        status = 200

        @staticmethod
        def read(_limit: int) -> bytes:
            return raw

    class Connection:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def request(self, *_args: object, **_kwargs: object) -> None:
            pass

        @staticmethod
        def getresponse() -> Response:
            return Response()

        def close(self) -> None:
            pass

    monkeypatch.setattr(runner.http.client, "HTTPConnection", Connection)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        runner._probe_capability_health(
            "http://127.0.0.1:8080",
            "test-token",
            timeout_s=1.0,
        )


def test_ds9_contract_rejects_non_loopback_and_short_shutdown() -> None:
    runner._validate_runtime_contract_args(_args())
    with pytest.raises(ValueError, match="loopback"):
        runner._validate_runtime_contract_args(_args(ws="ws://192.168.1.10:6008"))
    with pytest.raises(ValueError, match="shutdown timeout"):
        runner._validate_runtime_contract_args(_args(shutdown_timeout_s=30.0))


def test_ds9_runner_passes_only_auth_file_path_to_every_network_client(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "gateway-token"
    secret = "never-place-this-secret-on-a-command-line"

    gates = runner._build_gates(_args(), auth_token_file=token_file)

    by_name = {name: command for name, command, *_rest in gates}
    assert "--auth-token-file" not in by_name["rtsp"]
    for name in (
        "webrtc",
        "reid",
        "semantic-observation",
        "bev",
        "bridge",
        "floorplan",
        "ma-depth",
        "zero-copy-stats",
        "zero-copy-rest",
    ):
        command = by_name[name]
        values = _cli_values(command)
        assert values["--auth-token-file"] == str(token_file)
        assert secret not in runner._cmd_text(command)


def test_ds9_behavior_bundle_uses_strict_identity_and_all_camera_floorplan_gates(
    tmp_path: Path,
) -> None:
    args = _args(
        identity_require_cross_camera=True,
        identity_require_open_set=True,
        output_dir=tmp_path,
    )
    gates = runner._build_gates(args, auth_token_file=tmp_path / "token")
    by_name = {name: command for name, command, *_rest in gates}

    identity = by_name["reid"]
    identity_values = _cli_values(identity)
    assert "DS9/scripts/ds9_identity_shadow_live_gate.py" in identity
    assert "scripts/reid_stable_id_smoke_test.py" not in identity
    assert identity_values["--rest"] == "http://127.0.0.1:8080"
    assert identity_values["--pipeline-config"] == "DS9/config/infer.yaml"
    assert identity_values["--out"] == str(
        tmp_path / "identity-open-set-occupied.json"
    )
    assert identity_values["--session-id"] == "live-runner-test"
    assert identity_values["--runtime-lane"] == "baseline"
    assert identity_values["--runtime-instance-id"] == "runtime-instance-test"
    assert identity_values["--runtime-run-id"] == "runtime-run-test"
    assert identity_values["--source-out"] == str(
        tmp_path / "identity-open-set-occupied-source.json"
    )
    assert "--require-cross-camera" in identity
    assert "--require-open-set" in identity

    semantic = by_name["semantic-observation"]
    semantic_values = _cli_values(semantic)
    assert "DS9/scripts/ds9_semantic_observation_smoke_test.py" in semantic
    assert semantic_values["--identity-evidence"] == "/private/identity-v2.jsonl"
    assert semantic_values["--pipeline-config"] == "DS9/config/infer.yaml"
    assert semantic_values["--out"] == str(tmp_path / "semantic-observation.json")
    assert semantic_values["--snapshot-out"] == str(
        tmp_path / "semantic-identity-evidence.jsonl"
    )
    assert semantic_values["--source-out"] == str(
        tmp_path / "semantic-observation-source.json"
    )
    assert semantic_values["--session-id"] == "live-runner-test"
    assert semantic_values["--runtime-lane"] == "baseline"

    floorplan = by_name["floorplan"]
    floorplan_values = _cli_values(floorplan)
    assert "DS9/scripts/ds9_floorplan_live_gate.py" in floorplan
    assert "scripts/floorplan_rpc_smoke_test.py" not in floorplan
    assert floorplan_values["--pipeline-config"] == "DS9/config/infer.yaml"
    assert floorplan_values["--cameras-config"] == "config/cameras.yaml"
    assert floorplan_values["--timeout-s"] == "75.0"
    assert floorplan_values["--out"] == str(tmp_path / "mapanything-depth-quality.json")
    assert floorplan_values["--source-out"] == str(
        tmp_path / "mapanything-depth-quality-source.json"
    )
    assert floorplan_values["--session-id"] == "live-runner-test"
    assert floorplan_values["--runtime-lane"] == "baseline"


def test_attached_runner_binds_canonical_reports_to_exact_supervisor_session(
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "evidence" / "attached-session" / "launcher"
    launcher.mkdir(parents=True, mode=0o700)
    launcher.chmod(0o700)
    plan = {
        "schema_version": 1,
        "contract": "noesis.ds9.canonical_runtime_container",
        "mode": "plan",
        "session_id": "attached-session",
        "runtime_lane": "baseline",
        "ready_for_explicit_run": True,
        "canonical_runtime": {
            "ports": dict(runner.CANONICAL_PORTS),
            "endpoints": dict(runner.CANONICAL_ENDPOINTS),
            "source_ids": ["0", "1", "2"],
        },
        "session_paths": {"launcher_evidence": str(launcher)},
    }
    plan_path = launcher / "launch-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.chmod(0o600)
    identity_path = launcher / "runtime-identity.json"
    identity = {
        "schema_version": 1,
        "contract": runner.RUNTIME_IDENTITY_CONTRACT,
        "contract_version": 1,
        "session_id": "attached-session",
        "runtime_lane": "baseline",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "health_generated_at_us": 1_000_000,
        "observed_at_utc": "2026-07-11T00:00:01Z",
        "endpoints": dict(runner.CANONICAL_ENDPOINTS),
    }
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    identity_path.chmod(0o600)
    args = _args(
        session_id="attached-session",
        no_spawn=True,
        ownership_evidence_dir=launcher,
        identity_evidence=launcher.parent / "runtime" / "identity_v2.jsonl",
        output_dir=launcher.parent / "behavior",
    )

    runner._validate_runtime_contract_args(args)
    assert args.supervisor_runtime_instance_id == "runtime-instance-test"
    assert args.supervisor_runtime_run_id == "runtime-run-test"

    plan_path.write_text(
        json.dumps(plan).replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )
    plan_path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runner._validate_runtime_contract_args(args)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.chmod(0o600)

    identity_path.write_text(
        json.dumps(identity).replace(
            '"schema_version": 1',
            '"schema_version": 1, "schema_version": 1',
            1,
        ),
        encoding="utf-8",
    )
    identity_path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runner._validate_runtime_contract_args(args)
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    identity_path.chmod(0o600)
    gates = runner._build_gates(args, auth_token_file=tmp_path / "token")
    by_name = {name: command for name, command, *_rest in gates}
    assert _cli_values(by_name["reid"])["--out"] == str(
        launcher / "identity-open-set-occupied.json"
    )
    assert _cli_values(by_name["reid"])["--source-out"] == str(
        launcher / "identity-open-set-occupied-source.json"
    )
    assert _cli_values(by_name["floorplan"])["--out"] == str(
        launcher / "mapanything-depth-quality.json"
    )
    assert _cli_values(by_name["floorplan"])["--source-out"] == str(
        launcher / "mapanything-depth-quality-source.json"
    )
    assert _cli_values(by_name["semantic-observation"])["--out"] == str(
        launcher / "semantic-observation.json"
    )
    assert _cli_values(by_name["semantic-observation"])["--snapshot-out"] == str(
        launcher / "semantic-identity-evidence.jsonl"
    )
    assert _cli_values(by_name["semantic-observation"])["--source-out"] == str(
        launcher / "semantic-observation-source.json"
    )

    exact_identity_evidence = args.identity_evidence
    args.identity_evidence = launcher.parent.parent / "foreign" / "identity_v2.jsonl"
    with pytest.raises(ValueError, match="exact sibling runtime file"):
        runner._validate_runtime_contract_args(args)
    args.identity_evidence = exact_identity_evidence

    for field, invalid in (
        ("ws", "ws://127.0.0.1:6009"),
        ("rest", "http://127.0.0.1:8081"),
        ("rtsp_url", "rtsp://127.0.0.1:8554/other"),
    ):
        original = getattr(args, field)
        setattr(args, field, invalid)
        with pytest.raises(ValueError, match="exact canonical endpoints"):
            runner._validate_runtime_contract_args(args)
        setattr(args, field, original)

    plan["session_paths"]["launcher_evidence"] = str(launcher.parent / "other")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.chmod(0o600)
    with pytest.raises(ValueError, match="session/lane"):
        runner._validate_runtime_contract_args(args)
    plan["session_paths"]["launcher_evidence"] = str(launcher)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path.chmod(0o600)

    args.session_id = "other-session"
    with pytest.raises(ValueError, match="session/lane"):
        runner._validate_runtime_contract_args(args)


def test_ds9_runner_embeds_truthful_identity_claim_statuses_in_summary_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "identity-open-set-occupied.json"
    source = tmp_path / "identity-open-set-occupied-source.json"
    monkeypatch.setattr(
        runner.identity_gate,
        "validate_sealed_identity_shadow_report",
        lambda *_args, **_kwargs: {
            "claims": {
                "runtime_shadow_health": {"status": "pass", "required": True},
                "tracker_subject_continuity": {
                    "status": "pass",
                    "required": True,
                },
                "cross_camera_assignment_continuity": {
                    "status": "not_observed",
                    "required": False,
                },
                "open_set_non_force": {
                    "status": "not_observed",
                    "required": False,
                },
                "semantic_accuracy": {
                    "status": "not_evaluated",
                    "required": False,
                },
                "public_authority": {"status": "blocked", "required": True},
            }
        },
    )
    result = runner.StepResult(
        name="reid",
        ok=True,
        returncode=0,
        duration_s=1.0,
        log_path="reid.log",
        command="identity gate",
        notes="exit_0",
    )

    runner._apply_identity_gate_report(
        result,
        report,
        source_path=source,
        session_id="live-runner-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        pipeline_config=Path("DS9/config/infer.yaml"),
        require_cross_camera=False,
        require_open_set=False,
    )

    assert result.ok is True
    assert "cross_camera=not_observed" in result.notes
    assert "open_set_non_force=not_observed" in result.notes
    assert "semantic_accuracy=not_evaluated" in result.notes
    assert "public_authority=blocked" in result.notes
    assert "exact_source_replay=pass" in result.notes


def test_ds9_runner_rejects_incomplete_or_public_floorplan_gate_report(
    tmp_path: Path,
) -> None:
    report = tmp_path / "floorplan-all-cameras.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "noesis.ds9.floorplan-live-gate",
                "contract_version": 1,
                "session_id": "live-runner-test",
                "runtime_lane": "baseline",
                "runtime_instance_id": "runtime-instance-test",
                "runtime_run_id": "runtime-run-test",
                "ok": True,
                "active_camera_count": 3,
                "validated_camera_count": 1,
                "all_active_cameras_validated": False,
                "cameras": [{"camera_id": "living-room"}],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    report.chmod(0o644)
    result = runner.StepResult(
        name="floorplan",
        ok=True,
        returncode=0,
        duration_s=1.0,
        log_path="floorplan.log",
        command="floorplan gate",
        notes="exit_0",
    )

    runner._apply_floorplan_gate_report(
        result,
        report,
        session_id="live-runner-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
    )

    assert result.ok is False
    assert "mode must be 0600" in result.notes


def _floorplan_v3_runner_report() -> dict[str, object]:
    fresh = {
        "camera_id": "living-room",
        "snapshot_age_s": 0.5,
        "snapshot_ts_us": 9_500_000,
        "snapshot_ref": "living-room/capture.zarr",
        "snapshot_id": "capture-1",
        "snapshot_content_sha256": "a" * 64,
        "floorplan_ts_us": 10_000_000,
        "point_count": 20,
        "grid_shape": [2, 2],
        "served_from_cache": False,
        "floorplan_contract_version": 10,
        "calibration_fingerprint": "b" * 64,
        "floorplan_payload_sha256": "c" * 64,
        "layer_sha256s": {
            "density": "d" * 64,
            "observed": "2" * 64,
            "unknown": "3" * 64,
            "height": "4" * 64,
            "height_agl": "5" * 64,
            "distance": "6" * 64,
        },
        "observation_meta": {
            "contract": "noesis.floorplan.observation.v1",
            "observed_definition": "one_or_more_valid_projected_depth_points",
            "unknown_definition": (
                "zero_valid_projected_depth_points_within_grid_bounds"
            ),
            "observed_cells": 3,
            "unknown_cells": 1,
            "total_cells": 4,
        },
        "inferred_walkable_present": False,
        "capture_event_evidence_sha256": "e" * 64,
        "fusion_evidence_sha256": "f" * 64,
        "capture_event_id": "capture-event-sha256:" + "1" * 64,
        "capture_event_source_snapshot_count": 2,
        "capture_event_rgb_status": "not_requested",
    }
    cache = {
        key: copy.deepcopy(value)
        for key, value in fresh.items()
        if key
        not in {
            "capture_event_evidence_sha256",
            "fusion_evidence_sha256",
            "capture_event_id",
            "capture_event_source_snapshot_count",
            "capture_event_rgb_status",
        }
    }
    cache["served_from_cache"] = True
    cache["identity_unchanged"] = True
    cache["payload_unchanged"] = True
    stable_active = {"contract": "noesis.active_floorplan.health", "healthy": True}
    stable_controller = {
        "contract": "noesis.capture_event_controller_health",
        "healthy": True,
        "counters": {"requests_total": 1},
    }
    health = {
        "bev_frame": "camera_local_ground_m",
        "bev_health": {"healthy": True},
        "active_floorplan_health": stable_active,
        "capture_event_controller_health": stable_controller,
    }
    return {
        "schema_version": 3,
        "contract": "noesis.ds9.floorplan-live-gate",
        "contract_version": 3,
        "session_id": "live-runner-test",
        "runtime_lane": "baseline",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
        "ok": True,
        "max_snapshot_age_s": 120.0,
        "configured_camera_count": 1,
        "validated_camera_count": 1,
        "all_configured_camera_floorplans_validated": True,
        "exact_capture_event_camera_count": 1,
        "cache_only_validated_camera_count": 1,
        "cache_only_zero_mutation": True,
        "bev_renderer_ready": True,
        "bev_active_camera_count": 0,
        "bev_inactive_ready_camera_count": 1,
        "bev_failed_camera_count": 0,
        "all_configured_cameras_bev_ready": True,
        "cameras": [fresh],
        "cache_only_cameras": [cache],
        "runtime_health": {
            "after_fresh": health,
            "after_cache_only": copy.deepcopy(health),
        },
        "source_evidence": {},
        "errors": [],
    }


@pytest.mark.parametrize(
    "mutation",
    ("v1", "cache_flag", "rgb", "cache_identity", "health_mutation"),
)
def test_ds9_runner_requires_floorplan_v3_exact_and_zero_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    report = _floorplan_v3_runner_report()
    if mutation == "v1":
        report["schema_version"] = 1
        report["contract_version"] = 1
    elif mutation == "cache_flag":
        report["cache_only_zero_mutation"] = False
    elif mutation == "rgb":
        report["cameras"][0]["capture_event_rgb_status"] = "invalid"
    elif mutation == "cache_identity":
        report["cache_only_cameras"][0]["snapshot_id"] = "substituted"
    else:
        report["runtime_health"]["after_cache_only"][
            "capture_event_controller_health"
        ]["counters"]["requests_total"] = 2
    path = tmp_path / "mapanything-depth-quality.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    path.chmod(0o600)
    result = runner.StepResult(
        name="floorplan",
        ok=True,
        returncode=0,
        duration_s=1.0,
        log_path="floorplan.log",
        command="floorplan gate",
        notes="exit_0",
    )
    runner._apply_floorplan_gate_report(
        result,
        path,
        session_id="live-runner-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
    )
    assert result.ok is False
    assert "floorplan_report_invalid" in result.notes


def test_ds9_runner_accepts_exactly_replayed_floorplan_v4_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.floorplan_gate.FLOORPLAN_CONTRACT_VERSION == 10
    path = tmp_path / "mapanything-depth-quality.json"
    source = tmp_path / "mapanything-depth-quality-source.json"
    calls: list[tuple[Path, Path, dict[str, str]]] = []

    def load_exact(report_path: Path, source_path: Path, **identity: str):
        calls.append((report_path, source_path, identity))
        return {
            "configured_camera_count": 3,
            "bev_active_camera_count": 1,
            "bev_inactive_ready_camera_count": 2,
        }

    monkeypatch.setattr(
        runner.floorplan_gate,
        "load_and_validate_sealed_authority",
        load_exact,
    )
    result = runner.StepResult(
        name="floorplan",
        ok=True,
        returncode=0,
        duration_s=1.0,
        log_path="floorplan.log",
        command="floorplan gate",
        notes="exit_0",
    )
    runner._apply_floorplan_gate_report(
        result,
        path,
        session_id="live-runner-test",
        runtime_lane="baseline",
        runtime_instance_id="runtime-instance-test",
        runtime_run_id="runtime-run-test",
        source_path=source,
    )
    assert result.ok is True
    assert "cache_only_zero_mutation=pass" in result.notes
    assert "exact_source_replay=pass" in result.notes
    assert calls == [
        (
            path,
            source,
            {
                "session_id": "live-runner-test",
                "runtime_lane": "baseline",
                "runtime_instance_id": "runtime-instance-test",
                "runtime_run_id": "runtime-run-test",
            },
        )
    ]


def test_ds9_runner_requires_complete_semantic_observation_report(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    report = tmp_path / "semantic-observation.json"
    snapshot_path = tmp_path / "semantic-identity-evidence.jsonl"
    source_path = tmp_path / "semantic-observation-source.json"
    collector, valid_payload = semantic_fixture._evaluate(
        semantic_fixture._tracking_payload()
    )
    source_document = semantic_fixture.gate._source_transcript_document(
        session_id=semantic_fixture.SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=semantic_fixture.INSTANCE_ID,
        runtime_run_id=semantic_fixture.RUN_ID,
        acquisition_started_at_us=semantic_fixture.ACQUISITION_STARTED_AT_US,
        acquisition_finished_at_us=semantic_fixture.ACQUISITION_FINISHED_AT_US,
        collector=collector,
    )
    source_raw = semantic_fixture.gate._encoded_source_transcript(source_document)
    snapshot = semantic_fixture._snapshot()
    for path, body in (
        (
            report,
            (json.dumps(valid_payload, indent=2, sort_keys=True) + "\n").encode(),
        ),
        (snapshot_path, snapshot.payload),
        (source_path, source_raw),
    ):
        path.write_bytes(body)
        path.chmod(0o600)
    result = runner.StepResult(
        name="semantic-observation",
        ok=True,
        returncode=0,
        duration_s=1.0,
        log_path="semantic.log",
        command="semantic gate",
        notes="exit_0",
    )

    runner._apply_semantic_observation_gate_report(
        result,
        report,
        session_id=semantic_fixture.SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=semantic_fixture.INSTANCE_ID,
        runtime_run_id=semantic_fixture.RUN_ID,
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        identity_evidence_path=Path(snapshot.source_path),
    )

    assert result.ok is True
    assert "bounded_identity_cohorts=1" in result.notes

    incomplete = copy.deepcopy(valid_payload)
    incomplete.pop("sample_cohort")
    report.write_text(
        json.dumps(incomplete),
        encoding="utf-8",
    )
    result.ok = True
    runner._apply_semantic_observation_gate_report(
        result,
        report,
        session_id=semantic_fixture.SESSION_ID,
        runtime_lane="v3dt",
        runtime_instance_id=semantic_fixture.INSTANCE_ID,
        runtime_run_id=semantic_fixture.RUN_ID,
        pipeline_config=REPO_ROOT / "DS9/config/infer_v3dt.yaml",
        identity_evidence_path=Path(snapshot.source_path),
    )
    assert result.ok is False
    assert "semantic observation report is not canonical JSON" in result.notes


def test_completed_gate_dispatch_routes_only_semantic_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(
        lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        output_dir=tmp_path,
        identity_evidence=tmp_path / "runtime" / "identity_v2.jsonl",
    )
    calls: list[tuple[str, Path, dict[str, object]]] = []

    def apply_identity(
        result: runner.StepResult,
        report_path: Path,
        *,
        session_id: str,
        runtime_lane: str,
        runtime_instance_id: str,
        runtime_run_id: str,
        **_extra: object,
    ) -> None:
        calls.append(
            (
                result.name,
                report_path,
                {
                    "session_id": session_id,
                    "runtime_lane": runtime_lane,
                    "runtime_instance_id": runtime_instance_id,
                    "runtime_run_id": runtime_run_id,
                },
            )
        )

    def apply_semantic(
        result: runner.StepResult,
        report_path: Path,
        *,
        session_id: str,
        runtime_lane: str,
        runtime_instance_id: str,
        runtime_run_id: str,
        pipeline_config: Path,
        identity_evidence_path: Path,
    ) -> None:
        calls.append(
            (
                result.name,
                report_path,
                {
                    "session_id": session_id,
                    "runtime_lane": runtime_lane,
                    "runtime_instance_id": runtime_instance_id,
                    "runtime_run_id": runtime_run_id,
                    "pipeline_config": pipeline_config,
                    "identity_evidence_path": identity_evidence_path,
                },
            )
        )

    monkeypatch.setattr(runner, "_apply_identity_gate_report", apply_identity)
    monkeypatch.setattr(
        runner,
        "_apply_semantic_observation_gate_report",
        apply_semantic,
    )
    for name in ("reid", "semantic-observation"):
        runner._apply_completed_behavior_gate_report(
            name,
            runner.StepResult(
                name=name,
                ok=True,
                returncode=0,
                duration_s=1.0,
                log_path=f"{name}.log",
                command=name,
                notes="exit_0",
            ),
            args=args,
            lane=runner.V3DT_LANE,
        )

    common = {
        "session_id": "live-runner-test",
        "runtime_lane": "v3dt",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
    }
    assert calls == [
        (
            "reid",
            tmp_path / "v3dt-identity-open-set-occupied.json",
            common,
        ),
        (
            "semantic-observation",
            tmp_path / "semantic-observation.json",
            {
                **common,
                "pipeline_config": Path("DS9/config/infer_v3dt.yaml"),
                "identity_evidence_path": tmp_path
                / "runtime"
                / "identity_v2.jsonl",
            },
        ),
    ]


def test_completed_gate_dispatch_covers_all_remaining_behavior_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def recorder(branch: str):
        def apply(result: runner.StepResult, report_path: Path, **_kwargs: object) -> None:
            calls.append((branch, report_path.name))

        return apply

    monkeypatch.setattr(runner, "_apply_floorplan_gate_report", recorder("floorplan"))
    monkeypatch.setattr(runner, "_apply_v3dt_world_gate_report", recorder("v3dt-world"))
    monkeypatch.setattr(
        runner,
        "_apply_wholebody_occupied_gate_report",
        recorder("wholebody-occupied"),
    )
    monkeypatch.setattr(
        runner,
        "_apply_wholebody_media_gate_report",
        recorder("wholebody-media"),
    )

    def completed(name: str) -> runner.StepResult:
        return runner.StepResult(
            name=name,
            ok=True,
            returncode=0,
            duration_s=1.0,
            log_path=f"{name}.log",
            command=name,
        )

    baseline_args = _args(output_dir=tmp_path / "baseline")
    runner._apply_completed_behavior_gate_report(
        "floorplan",
        completed("floorplan"),
        args=baseline_args,
        lane=runner.BASELINE_LANE,
    )
    v3dt_args = _args(
        lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        output_dir=tmp_path / "v3dt",
    )
    runner._apply_completed_behavior_gate_report(
        "v3dt-world",
        completed("v3dt-world"),
        args=v3dt_args,
        lane=runner.V3DT_LANE,
    )
    wholebody_args = _args(
        lane="wholebody49-s",
        output_dir=tmp_path / "wholebody",
    )
    for name in ("wholebody-occupied", "webrtc"):
        runner._apply_completed_behavior_gate_report(
            name,
            completed(name),
            args=wholebody_args,
            lane=runner.WHOLEBODY49_S_LANE,
        )

    assert calls == [
        ("floorplan", runner.FLOORPLAN_REPORT_FILENAME),
        ("v3dt-world", runner.V3DT_WORLD_REPORT_FILENAME),
        ("wholebody-occupied", runner.WHOLEBODY49_REPORT_FILENAME),
        ("wholebody-media", runner.WHOLEBODY49_MEDIA_REPORT_FILENAME),
    ]


def test_new_behavior_apply_helpers_append_exact_replay_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.v3dt_world_gate,
        "validate_sealed_v3dt_world_report",
        lambda *_args, **_kwargs: {"tracks_seen": 6},
    )
    monkeypatch.setattr(
        runner.wholebody_gate,
        "validate_sealed_wholebody49_report",
        lambda *_args, **_kwargs: {"tracks_seen": 3},
    )
    monkeypatch.setattr(
        runner.wholebody_media_gate,
        "validate_sealed_wholebody49_media_report",
        lambda *_args, **_kwargs: {"metrics": {"decoded_frames": 4}},
    )

    def result(name: str) -> runner.StepResult:
        return runner.StepResult(
            name=name,
            ok=True,
            returncode=0,
            duration_s=1.0,
            log_path=f"{name}.log",
            command=name,
        )

    common = {
        "session_id": "live-runner-test",
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
    }
    world_result = result("v3dt-world")
    runner._apply_v3dt_world_gate_report(
        world_result,
        tmp_path / runner.V3DT_WORLD_REPORT_FILENAME,
        **common,
        runtime_lane="v3dt",
        pipeline_config=Path("DS9/config/infer_v3dt.yaml"),
        cameras_config=Path("DS9/config/cameras_v3dt.yaml"),
        launcher_dir=tmp_path,
    )
    occupied_result = result("wholebody-occupied")
    runner._apply_wholebody_occupied_gate_report(
        occupied_result,
        tmp_path / runner.WHOLEBODY49_REPORT_FILENAME,
        **common,
        runtime_lane="wholebody49-s",
        mode="masks",
        expected_source_ids=(0, 1, 2),
    )
    media_result = result("webrtc")
    runner._apply_wholebody_media_gate_report(
        media_result,
        tmp_path / runner.WHOLEBODY49_MEDIA_REPORT_FILENAME,
        **common,
        runtime_lane="wholebody49-s",
    )

    for item in (world_result, occupied_result, media_result):
        assert item.ok is True
        assert "exact_source_replay=pass" in item.notes


@pytest.mark.parametrize(
    ("ok", "returncode", "timed_out"),
    ((False, 1, False), (True, 0, True), (True, None, False)),
)
def test_completed_gate_dispatch_never_validates_failed_or_timed_out_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ok: bool,
    returncode: int | None,
    timed_out: bool,
) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_apply_identity_gate_report", forbidden)
    runner._apply_completed_behavior_gate_report(
        "reid",
        runner.StepResult(
            name="reid",
            ok=ok,
            returncode=returncode,
            duration_s=1.0,
            log_path="reid.log",
            command="reid",
            timed_out=timed_out,
        ),
        args=_args(output_dir=tmp_path),
        lane=runner.BASELINE_LANE,
    )
    assert called is False


def test_ds9_runner_token_load_is_read_only(tmp_path: Path) -> None:
    missing = tmp_path / "missing-token"

    with pytest.raises(Exception, match="missing"):
        runner._load_auth_token(missing)

    assert not missing.exists()


def test_ds9_shutdown_uses_graceful_sigterm_and_exact_zero(tmp_path: Path) -> None:
    code = """
import signal
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(_clean_shutdown_log(), encoding="utf-8")

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=2.0)

    assert result.ok is True
    assert result.returncode == 0
    assert result.forced_termination is False
    assert result.signal_sent is True
    assert result.command == "SIGTERM DS9 runtime"


def test_ds9_owned_shutdown_requires_every_orderly_lifecycle_marker(
    tmp_path: Path,
) -> None:
    code = """
import signal
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        _clean_shutdown_log().replace(
            "INFO pyservicemaker wait() returned (pipeline stopped)\n", ""
        ),
        encoding="utf-8",
    )

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=2.0)

    assert result.returncode == 0
    assert result.ok is False
    assert "servicemaker_wait_returned" in result.notes


@pytest.mark.parametrize(
    "failure_signature",
    (
        "PYSERVICEMAKER WAIT TIMEOUT",
        "pyservicemaker wait-timeout",
        "NATIVE PIPELINE TEARDOWN FAILED",
        "segmentation fault",
        "ERROR runtime teardown failed",
        "GStreamer-CRITICAL gst_object_unref assertion failed",
        "free(): INVALID POINTER",
    ),
)
def test_ds9_owned_shutdown_rejects_case_insensitive_wait_and_native_failures(
    tmp_path: Path, failure_signature: str
) -> None:
    code = """
import signal
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        _clean_shutdown_log() + failure_signature + "\n", encoding="utf-8"
    )

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=2.0)

    assert result.returncode == 0
    assert result.ok is False
    assert result.signatures


def test_ds9_owned_shutdown_rejects_out_of_order_lifecycle_evidence(
    tmp_path: Path,
) -> None:
    code = """
import signal
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "\n".join(
            (
                "INFO pyservicemaker wait() returned (pipeline stopped)",
                "INFO EOS received on pipeline (reason=shutdown_requested)",
                "INFO Orderly pipeline EOS accepted: component=orderly_eos_control request_sequence=1",
                "INFO Shutdown complete",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=2.0)

    assert result.returncode == 0
    assert result.ok is False
    assert "shutdown_eos_callback_before_servicemaker_wait_returned" in result.notes


def test_ds9_owned_shutdown_requires_exact_lifecycle_marker_order(
    tmp_path: Path,
) -> None:
    code = """
import signal
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text(
        "\n".join(
            (
                "INFO EOS received on pipeline (reason=shutdown_requested)",
                "INFO Orderly pipeline EOS accepted: component=orderly_eos_control request_sequence=1",
                "INFO pyservicemaker wait() returned (pipeline stopped)",
                "INFO Shutdown complete",
                "",
            )
        ),
        encoding="utf-8",
    )

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=2.0)

    assert result.returncode == 0
    assert result.ok is False
    assert "orderly_eos_accepted_before_shutdown_eos_callback" in result.notes


def test_ds9_forced_shutdown_is_a_failed_gate(tmp_path: Path) -> None:
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
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text("", encoding="utf-8")
    try:
        result = runner._shutdown_runtime(proc, runtime_log, timeout_s=0.1)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=2.0)

    assert result.ok is False
    assert result.returncode == -signal.SIGKILL
    assert result.forced_termination is True
    assert "SIGKILL" in result.notes


def test_ds9_premature_zero_exit_is_not_clean_orchestrated_shutdown(
    tmp_path: Path,
) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])
    assert proc.wait(timeout=2.0) == 0
    runtime_log = tmp_path / "runtime.log"
    runtime_log.write_text("", encoding="utf-8")

    result = runner._shutdown_runtime(proc, runtime_log, timeout_s=1.0)

    assert result.ok is False
    assert result.returncode == 0
    assert result.signal_sent is False
    assert "before orchestrated shutdown" in result.notes
