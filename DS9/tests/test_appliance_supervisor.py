from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from noesis_core.appliance import (
    ApplianceConfigurationError,
    DeploymentHealthBinding,
    runtime_context_environment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/run_canonical_runtime_container.py"
SPEC = importlib.util.spec_from_file_location("ds9_appliance_supervisor_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def _binding() -> DeploymentHealthBinding:
    return DeploymentHealthBinding(
        deployment_id="deploy-alpha",
        selector_sha256="a" * 64,
        state_release_id="release-alpha",
        runtime_family="ds9",
        runtime_variant="ds9:baseline",
        software_revision="b" * 40,
        boot_id="00000000-0000-0000-0000-000000000001",
    )


def _selected_state_binding(tmp_path: Path) -> SimpleNamespace:
    state_root = tmp_path / "selected-state"
    analytics = state_root / "payload" / "analytics"
    return SimpleNamespace(
        state_release_id="release-alpha",
        state=SimpleNamespace(
            build_directory=state_root / "runtime" / "noesis-build",
            runtime_files={
                "analytics_config": analytics / "nvdsanalytics.yaml",
                "analytics_exclude": (
                    analytics / "config_nvdsanalytics_exclude.ini"
                ),
                "identity_store": state_root / "payload" / "identity.db",
                "world_store": state_root / "payload" / "world.db",
            },
        ),
    )


def test_appliance_subcommands_accept_only_selector_and_digest() -> None:
    parsed = runtime._parse_args(
        [
            "appliance-check",
            "--deployment-selector",
            "/private/selector.json",
            "--selector-sha256",
            "a" * 64,
        ]
    )
    assert parsed.mode == "appliance-check"
    assert parsed.deployment_selector == Path("/private/selector.json")
    with pytest.raises(SystemExit):
        runtime._parse_args(["appliance-run", "--lane", "baseline"])


def test_appliance_context_is_explicitly_injected_into_canonical_container_command(
    tmp_path: Path,
) -> None:
    roots = runtime.HostRoots(
        docker=tmp_path / "docker",
        artifacts=tmp_path / "artifacts",
        runtime=tmp_path / "runtime",
    )
    secrets = runtime.SecretFiles(
        cameras=tmp_path / "camera.json",
        mapanything=tmp_path / "map.key",
        internal_auth=tmp_path / "token",
    )
    session = runtime.SessionPaths.from_root(roots.runtime, "appliance-test")
    context = runtime_context_environment(_binding())
    command = runtime.build_container_command(
        roots,
        secrets,
        session,
        runtime.BASELINE_LANE,
        appliance_context_env=context,
    )
    for name, value in context.items():
        assert f"{name}={value}" in command
    assert command[-len(runtime.canonical_runtime_arguments()) :] == (
        runtime.canonical_runtime_arguments()
    )


def test_pre_exec_snapshot_revalidation_rejects_revision_or_digest_drift() -> None:
    selector = SimpleNamespace(
        noesis_checkout=SimpleNamespace(
            software_revision="a" * 40,
            snapshot_sha256="b" * 64,
        )
    )
    binding = SimpleNamespace(selector=selector)
    with mock.patch.object(
        runtime,
        "compute_noesis_checkout_identity",
        return_value=SimpleNamespace(
            revision="a" * 40,
            snapshot_sha256="c" * 64,
        ),
    ):
        with pytest.raises(ApplianceConfigurationError, match="immediately before"):
            runtime.require_appliance_runtime_snapshot_current(binding)


def test_container_builder_rejects_partial_appliance_context(
    tmp_path: Path,
) -> None:
    roots = runtime.HostRoots(
        docker=tmp_path / "docker",
        artifacts=tmp_path / "artifacts",
        runtime=tmp_path / "runtime",
    )
    secrets = runtime.SecretFiles(
        cameras=tmp_path / "camera.json",
        mapanything=tmp_path / "map.key",
        internal_auth=tmp_path / "token",
    )
    session = runtime.SessionPaths.from_root(roots.runtime, "appliance-test")
    with pytest.raises(runtime.RuntimeContainerError, match="incomplete"):
        runtime.build_container_command(
            roots,
            secrets,
            session,
            appliance_context_env={"NOESIS_APPLIANCE_RUNTIME_CONTEXT": "{}"},
        )


def test_appliance_container_mounts_only_selected_release_state(
    tmp_path: Path,
) -> None:
    roots = runtime.HostRoots(
        docker=tmp_path / "docker",
        artifacts=tmp_path / "artifacts",
        runtime=tmp_path / "runtime",
    )
    secrets = runtime.SecretFiles(
        cameras=tmp_path / "camera.json",
        mapanything=tmp_path / "map.key",
        internal_auth=tmp_path / "token",
    )
    session = runtime.SessionPaths.from_root(roots.runtime, "appliance-test")
    binding = _selected_state_binding(tmp_path)

    command = runtime.build_container_command(
        roots,
        secrets,
        session,
        runtime.BASELINE_LANE,
        appliance_context_env=runtime_context_environment(_binding()),
        appliance_binding=binding,
    )
    mounts = {
        command[index + 1]
        for index, argument in enumerate(command[:-1])
        if argument == "--mount"
    }
    expected_writable = {
        runtime._mount(source, target, readonly=False)
        for source, target in runtime._runtime_writable_mounts(session, binding)
    }
    assert expected_writable <= mounts
    assert all(str(session.build) not in mount for mount in mounts)
    assert all(str(session.persistent_analytics) not in mount for mount in mounts)
    assert any(
        f"src={binding.state.runtime_files['world_store']},"
        f"dst={runtime.CONTAINER_STATE_ROOT / 'world_ds9.sqlite3'}" in mount
        for mount in mounts
    )
    assert any(
        f"src={binding.state.runtime_files['identity_store']},"
        f"dst={runtime.CONTAINER_STATE_ROOT / 'household' / 'identity_v2.sqlite3'}"
        in mount
        for mount in mounts
    )


def test_appliance_analytics_plan_prohibits_seed_or_copy(tmp_path: Path) -> None:
    roots = runtime.HostRoots(
        docker=tmp_path / "docker",
        artifacts=tmp_path / "artifacts",
        runtime=tmp_path / "runtime",
    )
    session = runtime.SessionPaths.from_root(roots.runtime, "appliance-test")
    binding = _selected_state_binding(tmp_path)

    contract = runtime.analytics_state_contract(
        session,
        runtime.BASELINE_LANE,
        binding,
    )

    assert contract["state_release_id"] == "release-alpha"
    assert contract["reviewed_seed"] is None
    assert contract["mount"]["source"] == str(
        binding.state.runtime_files["analytics_config"].parent
    )
    assert "without_seed_copy_or_fallback" in contract["initialization"]


def test_appliance_analytics_uses_selected_pair_without_seed_or_copy(
    tmp_path: Path,
) -> None:
    roots = runtime.HostRoots(
        docker=tmp_path / "docker",
        artifacts=tmp_path / "artifacts",
        runtime=tmp_path / "runtime",
    )
    session = runtime.SessionPaths.from_root(roots.runtime, "appliance-test")
    binding = _selected_state_binding(tmp_path)
    analytics = binding.state.runtime_files["analytics_config"].parent
    analytics.mkdir(parents=True, mode=0o700)
    analytics.chmod(0o700)
    config_payload = runtime._read_reviewed_analytics_seed()
    source_ids = runtime._load_lane_source_ids(runtime.BASELINE_LANE)
    parsed = runtime._load_analytics_config(
        config_payload,
        label="selected appliance analytics",
        canonical_source_ids=source_ids,
    )
    exclude_payload = runtime._render_analytics_exclude_ini(
        parsed,
        private_directory=analytics,
    )
    for path, payload in (
        (binding.state.runtime_files["analytics_config"], config_payload),
        (binding.state.runtime_files["analytics_exclude"], exclude_payload),
        (analytics / "release-metadata.json", b"{}\n"),
    ):
        path.write_bytes(payload)
        path.chmod(0o600)

    with mock.patch.object(
        runtime,
        "_read_reviewed_analytics_seed",
        side_effect=AssertionError("appliance mode attempted checkout seed"),
    ):
        evidence = runtime.prepare_analytics_state(
            session,
            runtime.BASELINE_LANE,
            binding,
        )

    assert evidence["action"] == "selected"
    assert evidence["persistence"] == "state_release"
    assert evidence["state_release_id"] == "release-alpha"
    assert evidence["config_path"] == str(
        binding.state.runtime_files["analytics_config"]
    )
