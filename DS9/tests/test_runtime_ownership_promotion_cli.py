from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


promotion = _load(
    REPO_ROOT / "DS9" / "scripts" / "promote_runtime_ownership_evidence.py",
    "runtime_ownership_promotion_cli_test_module",
)
registry = _load(
    REPO_ROOT / "DS9" / "scripts" / "runtime_ownership_registry.py",
    "runtime_ownership_promotion_registry_test_module",
)


def _fake_validator(*, status: str = "parity") -> tuple[ModuleType, list[object]]:
    module = ModuleType("runtime_ownership_promotion_fake_validator")
    matrix_raw = b"fixture ownership matrix\n"
    selector = {
        "profiles": ["canonical"],
        "realization_sha256": "1" * 64,
        "base_manifest_sha256": "2" * 64,
        "source_contracts_sha256": "3" * 64,
        "runtime_image_id": "sha256:" + ("6" * 64),
        "artifact_ids": ["engine.fixture"],
        "output_sha256": {"engine.fixture": "4" * 64},
    }
    released: list[object] = []
    supervisor = ModuleType("runtime_ownership_promotion_fake_supervisor")
    supervisor.ARTIFACT_TRANSACTION_LOCK_FILENAME = ".artifact.lock"
    supervisor.acquire_artifact_transaction_lock = lambda root: (
        73,
        Path(root) / ".artifact.lock",
    )
    supervisor.release_artifact_transaction_lock = lambda descriptor: released.append(
        descriptor
    )
    sys.modules[supervisor.__name__] = supervisor

    module.EXPECTED_MATRIX_ID = "noesis-ds8-ds9-runtime-ownership"
    module.STRICT_BLOCKING_STATUSES = {"known_gap", "blocked"}
    module.ASSET_REALIZATION_KEYS = frozenset(selector)
    module.CAPABILITY_REGISTRY = {
        "artifacts.canonical_graph": {
            "surface": "artifacts",
            "minimum_tier": "asset_realization",
            "required_profiles": ["canonical"],
        }
    }
    module._load_yaml_with_raw = lambda _path: (
        {
            "schema_version": 2,
            "matrix_id": module.EXPECTED_MATRIX_ID,
            "capabilities": [
                {
                    "id": "artifacts.canonical_graph",
                    "surface": "artifacts",
                    "status": status,
                    "evidence": {"repository_source": {"ds9": {"path": "x"}}},
                }
            ],
        },
        matrix_raw,
    )
    module._sha256_bytes = lambda raw: hashlib.sha256(raw).hexdigest()
    module._current_checkout_summary = lambda: {"sha256": "5" * 64}
    module._require_sha256 = lambda value, _label: value
    module.build_asset_realization_selector = lambda **_kwargs: (
        dict(selector),
        dict(selector),
    )
    module._active_registry_promotions = lambda snapshot, **kwargs: (
        {
            key[0]: {key[1]: {key[2]: dict(event["selector"])}}
            for key, event in snapshot.active_for_matrix(
                kwargs["matrix_id"], kwargs["matrix_sha256"]
            ).items()
        },
        {
            key[0]: {key[1]: {key[2]: dict(event)}}
            for key, event in snapshot.active_for_matrix(
                kwargs["matrix_id"], kwargs["matrix_sha256"]
            ).items()
        },
    )

    def validate_matrix(_matrix, *, promoted_evidence, **_kwargs):
        selected = promoted_evidence["artifacts.canonical_graph"][
            "asset_realization"
        ]["ds9"]
        return {
            "ok": True,
            "errors": [],
            "evidence_details": {
                "artifacts.canonical_graph": {
                    "asset_realization": {"ds9": dict(selected)}
                }
            },
        }

    module.validate_matrix = validate_matrix
    module._load_runtime_supervisor_module = lambda: supervisor
    module._validate_asset_realization_evidence = lambda *_args, **_kwargs: dict(
        selector
    )
    module._strict_json_equal = lambda left, right: left == right
    return module, released


def test_promotion_command_appends_and_explicitly_supersedes(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    artifact_root = tmp_path / "artifacts"
    runtime_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    runtime_root.chmod(0o700)
    artifact_root.chmod(0o700)
    validator, released = _fake_validator()

    def loader(path: Path, _name: str):
        return validator if path == promotion.VALIDATOR_PATH else registry

    args = promotion._parse_args(
        [
            "--runtime-root",
            str(runtime_root),
            "--artifact-root",
            str(artifact_root),
            "--capability-id",
            "artifacts.canonical_graph",
            "--evidence-type",
            "asset_realization",
            "--subject",
            "ds9",
        ]
    )
    with mock.patch.object(promotion, "_load_module", side_effect=loader):
        first = promotion.promote(args)
    assert first["ok"] is True
    assert released == [73]

    args.supersedes_event_digest = first["event"]["event_digest"]
    validator, released_second = _fake_validator()
    with mock.patch.object(
        promotion,
        "_load_module",
        side_effect=lambda path, _name: (
            validator if path == promotion.VALIDATOR_PATH else registry
        ),
    ):
        second = promotion.promote(args)
    assert second["event"]["sequence"] == 2
    assert second["event"]["supersedes_event_digest"] == first["event"][
        "event_digest"
    ]
    assert released_second == [73]
    assert len(registry.read_registry(runtime_root).events) == 2


def test_promotion_command_refuses_normative_blocker(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    artifact_root = tmp_path / "artifacts"
    runtime_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    validator, _released = _fake_validator(status="blocked")
    args = promotion._parse_args(
        [
            "--runtime-root",
            str(runtime_root),
            "--artifact-root",
            str(artifact_root),
            "--capability-id",
            "artifacts.canonical_graph",
            "--evidence-type",
            "asset_realization",
            "--subject",
            "ds9",
        ]
    )
    with (
        mock.patch.object(
            promotion,
            "_load_module",
            side_effect=lambda path, _name: (
                validator if path == promotion.VALIDATOR_PATH else registry
            ),
        ),
        pytest.raises(ValueError, match="cannot override"),
    ):
        promotion.promote(args)


def test_asset_promotion_terminal_cas_rejects_runtime_image_drift(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    artifact_root = tmp_path / "artifacts"
    runtime_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)
    validator, released = _fake_validator()
    selector, _detail = validator.build_asset_realization_selector()
    drifted = dict(selector)
    drifted["runtime_image_id"] = "sha256:" + ("e" * 64)
    validator._validate_asset_realization_evidence = (
        lambda *_args, **_kwargs: dict(drifted)
    )
    args = promotion._parse_args(
        [
            "--runtime-root",
            str(runtime_root),
            "--artifact-root",
            str(artifact_root),
            "--capability-id",
            "artifacts.canonical_graph",
            "--evidence-type",
            "asset_realization",
            "--subject",
            "ds9",
        ]
    )

    with (
        mock.patch.object(
            promotion,
            "_load_module",
            side_effect=lambda path, _name: (
                validator if path == promotion.VALIDATOR_PATH else registry
            ),
        ),
        pytest.raises(ValueError, match="artifact changed before registry append"),
    ):
        promotion.promote(args)

    assert released == [73]
    assert registry.read_registry(runtime_root).events == ()


def test_runtime_promotion_requires_terminal_checksum_covered_cohort_cas(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    artifact_root = tmp_path / "artifacts"
    docker_root = tmp_path / "docker"
    for root in (runtime_root, artifact_root, docker_root):
        root.mkdir(mode=0o700)
        root.chmod(0o700)

    validator, released = _fake_validator()
    matrix_raw = b"fixture runtime ownership matrix\n"
    checksum_raw = b"0" * 64 + b"  summary.json\n"
    checksum_sha256 = hashlib.sha256(checksum_raw).hexdigest()
    asset_selector, _asset_detail = validator.build_asset_realization_selector()
    selector = {
        **asset_selector,
        "session_id": "v3dt-session-test",
        "lane": "v3dt",
        "checksum_manifest": "launcher/SHA256SUMS",
        "checksum_sha256": checksum_sha256,
        "behavior_documents": ["v3dt_world_gate_v2"],
    }
    detail = {
        **selector,
        "runtime_instance_id": "runtime-instance-test",
        "runtime_run_id": "runtime-run-test",
    }
    validator.CAPABILITY_REGISTRY = {
        "tracking.v3dt": {
            "surface": "tracking",
            "minimum_tier": "runtime_session",
            "required_profiles": ["canonical"],
            "runtime_requirements": {
                "v3dt": ["v3dt_world_gate_v2"],
            },
        }
    }
    validator._load_yaml_with_raw = lambda _path: (
        {
            "schema_version": 2,
            "matrix_id": validator.EXPECTED_MATRIX_ID,
            "capabilities": [
                {
                    "id": "tracking.v3dt",
                    "surface": "tracking",
                    "status": "parity",
                    "evidence": {"repository_source": {"ds9": {"path": "x"}}},
                }
            ],
        },
        matrix_raw,
    )
    validator.build_runtime_session_selector = lambda **_kwargs: (
        dict(selector),
        dict(detail),
    )

    def validate_matrix(_matrix, *, promoted_evidence, **_kwargs):
        selected = promoted_evidence["tracking.v3dt"]["runtime_session"][
            "v3dt-session"
        ]
        return {
            "ok": True,
            "errors": [],
            "evidence_details": {
                "tracking.v3dt": {
                    "runtime_session": {
                        "v3dt-session": {**detail, **dict(selected)}
                    }
                }
            },
        }

    validator.validate_matrix = validate_matrix
    validator._external_root = lambda root, *_args, **_kwargs: Path(root)
    validator._anchored_file_content = lambda *_args, **_kwargs: checksum_raw
    terminal_cohort_checks: list[Path] = []

    def reject_terminal_cohort(
        _runtime: Path,
        launcher_relative: Path,
        *,
        expected_manifest_sha256: str,
    ) -> dict[str, bytes]:
        assert expected_manifest_sha256 == checksum_sha256
        terminal_cohort_checks.append(launcher_relative)
        raise ValueError("covered cohort drift")

    validator._load_checksum_covered_files = reject_terminal_cohort
    args = promotion._parse_args(
        [
            "--runtime-root",
            str(runtime_root),
            "--artifact-root",
            str(artifact_root),
            "--docker-root",
            str(docker_root),
            "--capability-id",
            "tracking.v3dt",
            "--evidence-type",
            "runtime_session",
            "--subject",
            "v3dt-session",
            "--lane",
            "v3dt",
            "--session-id",
            "v3dt-session-test",
        ]
    )

    with (
        mock.patch.object(
            promotion,
            "_load_module",
            side_effect=lambda path, _name: (
                validator if path == promotion.VALIDATOR_PATH else registry
            ),
        ),
        pytest.raises(ValueError, match="covered cohort drift"),
    ):
        promotion.promote(args)

    assert terminal_cohort_checks == [
        Path("evidence/v3dt-session-test/launcher")
    ]
    assert released == [73]
    assert registry.read_registry(runtime_root).events == ()
