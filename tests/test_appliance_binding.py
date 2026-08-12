from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

import noesis_core.appliance as appliance_contract
from noesis_core.appliance import (
    ApplianceConfigurationError,
    canonical_json,
    compute_noesis_checkout_identity,
    load_appliance_binding,
    NOESIS_RUNTIME_DIRECTORY_ROOTS,
    NOESIS_RUNTIME_DIRECT_FILES,
    NOESIS_RUNTIME_EXTENSION_MODULES,
    NOESIS_RUNTIME_MODEL_FILES,
    require_appliance_state_current,
)
from noesis_core.contracts.appliance import (
    ApplianceEndpoints,
    ApplianceReadinessVersions,
    CheckoutBinding,
    DeploymentSelector,
    DS8RuntimeSelector,
    StateBaseline,
    StateBaselineBinding,
    StateBaselineFile,
    StateMigration,
    StateRelease,
    StateReleaseBinding,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REVISION = "d" * 40


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _private_contract(path: Path, value: object) -> bytes:
    payload = canonical_json(value)
    path.write_bytes(payload)
    path.chmod(0o600)
    return payload


def _fixture(
    tmp_path: Path,
    *,
    extra_state_payloads: dict[str, bytes] | None = None,
) -> tuple[Path, str, Path, dict[str, str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_path.chmod(0o700)
    noesis_root = _private_directory(tmp_path / "noesis-release")
    menon_root = _private_directory(tmp_path / "menon-release")
    release_root = _private_directory(tmp_path / "state-release")
    build_root = _private_directory(release_root / "runtime" / "noesis-build")
    selector_root = _private_directory(tmp_path / "selectors")
    state_payloads = {
        "payload/analytics/nvdsanalytics.yaml": b"analytics:\n  stages: {}\n",
        "payload/analytics/config_nvdsanalytics_exclude.ini": b"[property]\nenable=1\n",
        "payload/identity.db": b"identity-state\n",
        "payload/scene/scene_releases.sqlite3": b"scene-state\n",
        "payload/world.db": b"world-state\n",
    }
    state_payloads.update(extra_state_payloads or {})
    baseline_files: dict[str, StateBaselineFile] = {}
    for relative, payload in state_payloads.items():
        file = release_root / relative
        _private_directory(file.parent)
        file.write_bytes(payload)
        file.chmod(0o600)
        baseline_files[relative] = StateBaselineFile(
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    baseline_path = release_root / "activation-baseline.json"
    baseline = StateBaseline(
        contract="noesis.appliance.state_baseline",
        contract_version=1,
        release_id="release-alpha",
        created_at="2026-07-11T12:00:00.000Z",
        files=baseline_files,
    )
    baseline_payload = _private_contract(baseline_path, baseline)
    manifest_path = release_root / "release.json"
    release = StateRelease(
        contract="noesis.appliance.state_release",
        contract_version=1,
        release_id="release-alpha",
        parent_release_id=None,
        created_at="2026-07-11T12:01:00.000Z",
        root=os.fspath(release_root),
        schemas={"analytics_roi": 1, "identity": 1, "scene": 1, "world": 3},
        migration=StateMigration(
            mode="fresh",
            tool_sha256=SHA_A,
            report_sha256=SHA_B,
        ),
        baseline=StateBaselineBinding(
            inventory_path=os.fspath(baseline_path),
            inventory_sha256=hashlib.sha256(baseline_payload).hexdigest(),
            file_count=len(baseline_files),
            byte_count=sum(row.bytes for row in baseline_files.values()),
        ),
    )
    manifest_payload = _private_contract(manifest_path, release)
    selector = DeploymentSelector(
        contract="noesis.appliance.deployment_selector",
        contract_version=1,
        deployment_id="deploy-alpha",
        created_at="2026-07-11T12:02:00.000Z",
        bundle_sha256=SHA_A,
        menon_checkout=CheckoutBinding(
            root=os.fspath(menon_root),
            snapshot_kind="menon-runtime-v1",
            snapshot_sha256=SHA_B,
            software_revision="e" * 40,
        ),
        noesis_checkout=CheckoutBinding(
            root=os.fspath(noesis_root),
            snapshot_kind="noesis-runtime-v1",
            snapshot_sha256=SHA_C,
            software_revision=REVISION,
        ),
        state_release=StateReleaseBinding(
            release_id="release-alpha",
            manifest_path=os.fspath(manifest_path),
            manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        ),
        endpoints=ApplianceEndpoints(
            websocket="ws://127.0.0.1:6008",
            rest="http://127.0.0.1:8080",
            rtsp="rtsp://127.0.0.1:8554/ds-test",
        ),
        runtime=DS8RuntimeSelector(
            family="ds8",
            pgie_profile="yolo26",
            model_size="m",
            tracking_mode="baseline",
        ),
        readiness=ApplianceReadinessVersions(
            capability_contract_version=1,
            deployment_health_contract_version=1,
            websocket_health_contract_version=2,
        ),
    )
    selector_payload = canonical_json(selector)
    selector_sha256 = hashlib.sha256(selector_payload).hexdigest()
    selector_path = selector_root / f"{selector_sha256}.json"
    _private_contract(selector_path, selector)
    env = {
        "NOESIS_DEPLOYMENT_ID": "deploy-alpha",
        "NOESIS_DEPLOYMENT_SELECTOR_FILE": os.fspath(selector_path),
        "NOESIS_DEPLOYMENT_SELECTOR_SHA256": selector_sha256,
        "NOESIS_STATE_RELEASE_ID": "release-alpha",
        "NOESIS_STATE_RELEASE_MANIFEST": os.fspath(manifest_path),
        "NOESIS_STATE_RELEASE_ROOT": os.fspath(release_root),
        "NOESIS_RUNTIME_FAMILY": "ds8",
        "NOESIS_ANALYTICS_CONFIG": os.fspath(
            release_root / "payload/analytics/nvdsanalytics.yaml"
        ),
        "NOESIS_ANALYTICS_EXCLUDE_CONFIG": os.fspath(
            release_root / "payload/analytics/config_nvdsanalytics_exclude.ini"
        ),
        "NOESIS_IDENTITY_V2_STORE": os.fspath(release_root / "payload/identity.db"),
        "NOESIS_SCENE_STORE_PATH": os.fspath(
            release_root / "payload/scene/scene_releases.sqlite3"
        ),
        "NOESIS_WORLD_JOURNAL_PATH": os.fspath(release_root / "payload/world.db"),
        "NOESIS_BUILD_DIR": os.fspath(build_root),
    }
    return selector_path, selector_sha256, noesis_root, env


def _load(tmp_path: Path, **overrides: object):
    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path)
    arguments: dict[str, object] = {
        "selector_file": selector_path,
        "selector_sha256": selector_sha256,
        "repo_root": noesis_root,
        "env": env,
        "expected_family": "ds8",
        "expected_profile": "yolo26",
        "expected_size": "m",
        "expected_tracking_mode": "baseline",
        "verify_checkout": False,
    }
    arguments.update(overrides)
    return load_appliance_binding(**arguments)


def test_exact_selector_release_and_environment_binding_is_accepted(tmp_path: Path) -> None:
    binding = _load(tmp_path)
    assert binding.deployment_id == "deploy-alpha"
    assert binding.runtime_variant == "ds8:yolo26:m:baseline"
    assert binding.state.release.schemas == {
        "analytics_roi": 1,
        "identity": 1,
        "scene": 1,
        "world": 3,
    }
    assert binding.selector.noesis_checkout.snapshot_kind == "noesis-runtime-v1"


def test_selector_rejects_wrong_runtime_tuple_and_inherited_binding(tmp_path: Path) -> None:
    with pytest.raises(ApplianceConfigurationError, match="arguments"):
        _load(tmp_path, expected_size="s")

    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path / "env")
    env["NOESIS_STATE_RELEASE_ID"] = "release-other"
    with pytest.raises(ApplianceConfigurationError, match="NOESIS_STATE_RELEASE_ID"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_selector_and_state_files_require_private_canonical_bytes(tmp_path: Path) -> None:
    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path)
    selector_path.chmod(0o644)
    with pytest.raises(ApplianceConfigurationError, match="mode-0600"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_selected_mutable_state_and_build_directory_require_private_modes(
    tmp_path: Path,
) -> None:
    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path)
    Path(env["NOESIS_WORLD_JOURNAL_PATH"]).chmod(0o644)
    with pytest.raises(ApplianceConfigurationError, match="writable mode-0600"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )

    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path / "build")
    Path(env["NOESIS_BUILD_DIR"]).chmod(0o755)
    with pytest.raises(ApplianceConfigurationError, match="build directory"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_selected_build_directory_environment_cannot_drift(tmp_path: Path) -> None:
    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path)
    env["NOESIS_BUILD_DIR"] = os.fspath(tmp_path / "unselected-build")
    with pytest.raises(ApplianceConfigurationError, match="NOESIS_BUILD_DIR"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_selected_state_file_substitution_is_rejected_before_runtime_use(
    tmp_path: Path,
) -> None:
    binding = _load(tmp_path)
    world = binding.state.runtime_files["world_store"]
    original = world.with_name("world.original")
    world.rename(original)
    world.write_bytes(original.read_bytes())
    world.chmod(0o600)

    with pytest.raises(ApplianceConfigurationError, match="world_store.*changed"):
        require_appliance_state_current(binding)


def test_selected_selector_file_substitution_is_rejected_before_runtime_use(
    tmp_path: Path,
) -> None:
    binding = _load(tmp_path)
    original = binding.selector_path.with_suffix(".original")
    binding.selector_path.rename(original)
    binding.selector_path.write_bytes(original.read_bytes())
    binding.selector_path.chmod(0o600)

    with pytest.raises(ApplianceConfigurationError, match="selector changed"):
        require_appliance_state_current(binding)


def test_noesis_rejects_nested_state_lease_environment(tmp_path: Path) -> None:
    selector_path, selector_sha256, noesis_root, env = _fixture(tmp_path)
    env["NOESIS_STATE_RELEASE_LEASE_FILE"] = os.fspath(tmp_path / "forbidden.lease")
    with pytest.raises(ApplianceConfigurationError, match="must not receive or acquire"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_ds9_selector_freezes_the_exact_checkout_supervisor(tmp_path: Path) -> None:
    selector_path, _selector_sha256, noesis_root, _env = _fixture(tmp_path)
    payload = json.loads(selector_path.read_text(encoding="utf-8"))
    runtime = {
        "family": "ds9",
        "lane": "baseline",
        "supervisor_path": os.fspath(noesis_root / "DS9" / "scripts" / "alternate.py"),
        "runtime_image_id": f"sha256:{SHA_A}",
        "build_image_id": f"sha256:{SHA_B}",
        "docker_root": os.fspath(tmp_path / "docker"),
        "artifact_root": os.fspath(tmp_path / "artifacts"),
        "runtime_root": os.fspath(tmp_path / "runtime"),
        "asset_realization_sha256": SHA_A,
        "ownership_matrix_sha256": SHA_B,
    }
    with pytest.raises(ValidationError, match="canonical runtime supervisor"):
        DeploymentSelector.model_validate({**payload, "runtime": runtime})

    runtime["supervisor_path"] = os.fspath(
        noesis_root / "DS9" / "scripts" / "run_canonical_runtime_container.py"
    )
    assert DeploymentSelector.model_validate({**payload, "runtime": runtime}).runtime.family == "ds9"


def test_runtime_build_content_cannot_enter_activation_baseline(tmp_path: Path) -> None:
    selector_path, selector_sha256, noesis_root, env = _fixture(
        tmp_path,
        extra_state_payloads={"runtime/noesis-build/cache.bin": b"cache\n"},
    )
    with pytest.raises(ApplianceConfigurationError, match="outside the state baseline"):
        load_appliance_binding(
            selector_file=selector_path,
            selector_sha256=selector_sha256,
            repo_root=noesis_root,
            env=env,
            expected_family="ds8",
            expected_profile="yolo26",
            expected_size="m",
            expected_tracking_mode="baseline",
            verify_checkout=False,
        )


def test_closed_models_reject_old_git_only_kind_and_coercive_counts() -> None:
    with pytest.raises(ValidationError):
        CheckoutBinding(
            root="/tmp/noesis",
            snapshot_kind="noesis-git-v1",
            snapshot_sha256=SHA_A,
            software_revision=REVISION,
        )
    with pytest.raises(ValidationError):
        StateBaselineFile(sha256=SHA_A, bytes="17")
    with pytest.raises(ValidationError):
        DS8RuntimeSelector(
            family="ds8",
            pgie_profile="wholebody49",
            model_size="m",
            tracking_mode="baseline",
        )


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", os.fspath(root), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _runtime_checkout(tmp_path: Path) -> tuple[Path, Path]:
    root = _private_directory(tmp_path / "runtime-checkout")
    for relative in NOESIS_RUNTIME_DIRECTORY_ROOTS:
        _private_directory(root / relative)
    for relative in NOESIS_RUNTIME_DIRECT_FILES:
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"parser\n")
        file.chmod(0o755)
    for module in NOESIS_RUNTIME_EXTENSION_MODULES:
        file = root / f"{module}.cpython-312-x86_64-linux-gnu.so"
        file.write_bytes(f"{module}\n".encode("ascii"))
        file.chmod(0o755)
    model_root = _private_directory(tmp_path / "external-model-store")
    (root / "models").symlink_to(model_root, target_is_directory=True)
    for relative in NOESIS_RUNTIME_MODEL_FILES:
        suffix = relative.removeprefix("models/")
        file = model_root / suffix
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(f"{relative}\n".encode("ascii"))
        file.chmod(0o600)
    (root / ".gitignore").write_text("*\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Noesis Test")
    _git(root, "config", "user.email", "noesis-test@example.invalid")
    _git(root, "add", "-f", ".gitignore")
    _git(root, "commit", "-qm", "runtime snapshot fixture")
    return root, model_root


def test_noesis_runtime_identity_captures_ignored_external_model_bytes(
    tmp_path: Path,
) -> None:
    root, model_root = _runtime_checkout(tmp_path)
    first = compute_noesis_checkout_identity(root)
    selected = model_root / NOESIS_RUNTIME_MODEL_FILES[0].removeprefix("models/")
    selected.write_bytes(b"changed ignored runtime bytes\n")
    second = compute_noesis_checkout_identity(root)
    assert second.revision == first.revision
    assert second.tree == first.tree
    assert second.runtime_inventory_sha256 != first.runtime_inventory_sha256
    assert second.snapshot_sha256 != first.snapshot_sha256


def test_noesis_runtime_identity_rejects_final_model_link_and_hardlink(
    tmp_path: Path,
) -> None:
    root, model_root = _runtime_checkout(tmp_path)
    relative = NOESIS_RUNTIME_MODEL_FILES[0].removeprefix("models/")
    selected = model_root / relative
    original = selected.with_name(selected.name + ".original")
    selected.rename(original)
    selected.symlink_to(original.name)
    with pytest.raises(ApplianceConfigurationError, match="invalid"):
        compute_noesis_checkout_identity(root)

    selected.unlink()
    os.link(original, selected)
    with pytest.raises(ApplianceConfigurationError, match="invalid"):
        compute_noesis_checkout_identity(root)


def test_noesis_runtime_identity_rejects_non_ascii_inventory_paths(
    tmp_path: Path,
) -> None:
    root, _model_root = _runtime_checkout(tmp_path)
    unportable = root / "config" / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.ini"
    unportable.write_bytes(b"runtime config\n")
    unportable.chmod(0o600)

    with pytest.raises(ApplianceConfigurationError, match="printable ASCII"):
        compute_noesis_checkout_identity(root)


def test_noesis_runtime_identity_rechecks_git_cohort_after_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _model_root = _runtime_checkout(tmp_path)
    original = appliance_contract._compute_noesis_runtime_inventory

    def mutate_tracked_source(checkout: Path) -> tuple[str, int, int]:
        result = original(checkout)
        (checkout / ".gitignore").write_text("changed during admission\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        appliance_contract,
        "_compute_noesis_runtime_inventory",
        mutate_tracked_source,
    )

    with pytest.raises(ApplianceConfigurationError, match="changed during"):
        compute_noesis_checkout_identity(root)
