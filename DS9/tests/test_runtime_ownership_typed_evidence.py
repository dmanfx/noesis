from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "DS9" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ownership = _load_script("validate_runtime_ownership")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _private_write(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def _private_json(path: Path, payload: object) -> None:
    _private_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


class TypedEvidenceFixture:
    session_id = "fixture-session"
    artifact_id = "engine.fixture"
    image_id = "sha256:" + ("b" * 64)
    container_id = "a" * 64

    def __init__(self, base: Path) -> None:
        self.repo = base / "repo"
        self.artifact_root = base / "artifacts"
        self.runtime_root = base / "runtime"
        self.docker_root = base / "docker"
        self.repo.mkdir(mode=0o755)
        self.artifact_root.mkdir(mode=0o750)
        self.runtime_root.mkdir(mode=0o700)
        self.docker_root.mkdir(mode=0o750)
        self.docker_root.chmod(0o750)
        (self.docker_root / "data").mkdir()
        (self.repo / "noesis").mkdir()
        (self.repo / "DS9" / "noesis").mkdir(parents=True)
        (self.repo / "DS9" / "scripts").mkdir(parents=True)
        (self.repo / "DS9" / "config").mkdir(parents=True)
        (self.repo / "config").mkdir()
        (self.repo / "source.py").write_text("MARKER = True\n", encoding="utf-8")
        (self.repo / "DS9" / "config" / "infer.yaml").write_text(
            yaml.safe_dump(
                {
                    "sources": [
                        {"uri_secret": "living-room"},
                        {"uri_secret": "kitchen"},
                        {"uri_secret": "family-room"},
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (self.repo / "config" / "cameras.yaml").write_text(
            yaml.safe_dump(
                {
                    "cameras": {
                        0: {"name": "living-room"},
                        1: {"name": "kitchen"},
                        2: {"name": "family-room"},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        self.output = self.artifact_root / "models" / "engines" / "fixture.engine"
        self.output.parent.mkdir(parents=True)
        (self.artifact_root / "models").chmod(0o755)
        self.output.parent.chmod(0o755)
        self.output.write_bytes(b"fixture-engine-bytes")
        self.output.chmod(0o644)
        self.output_sha256 = _sha256(self.output.read_bytes())

        self.manifest_path = self.repo / "DS9" / "asset_manifest.yaml"
        self.manifest = {
            "policy": {"profile_inheritance": {"canonical": ["runtime_common"]}},
            "runtime": {"image": {"image_id": self.image_id}},
            "artifacts": [
                {
                    "id": self.artifact_id,
                    "kind": "tensorrt_engine",
                    "output": "DS9/models/engines/fixture.engine",
                    "required_profiles": ["canonical"],
                }
            ],
        }
        self.manifest_path.write_text(
            yaml.safe_dump(self.manifest, sort_keys=False), encoding="utf-8"
        )
        self.manifest_sha256 = _sha256(self.manifest_path.read_bytes())

        self.source_contracts_path = (
            self.repo / "DS9" / "config" / "engine_source_contracts.json"
        )
        self.source_contracts_path.write_text(
            json.dumps({"schema_version": 1, "contracts": {}}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.source_contracts_sha256 = _sha256(
            self.source_contracts_path.read_bytes()
        )

        self.realization_path = self.artifact_root / "asset_realization.json"
        self.realization = {
            "schema_version": 1,
            "contract": ownership.REALIZATION_CONTRACT,
            "base_manifest": {
                "path": "DS9/asset_manifest.yaml",
                "sha256": self.manifest_sha256,
            },
            "source_contracts": {
                "path": "DS9/config/engine_source_contracts.json",
                "sha256": self.source_contracts_sha256,
            },
            "created_at_utc": "2026-07-11T00:00:00Z",
            "updated_at_utc": "2026-07-11T00:00:00Z",
            "artifacts": {
                self.artifact_id: {
                    "state": "staged_unverified",
                    "provenance": {"output_sha256": self.output_sha256},
                }
            },
        }
        self.write_realization()

        self.snapshot = {
            "sha256": "c" * 64,
            "file_count": 12,
            "byte_count": 3456,
            "manifest_binary_count": 1,
            "manifest_binary_paths": ["DS9/native_extensions/fixture.so"],
        }
        self.gpu = {
            "compute_capability": "8.6",
            "driver_version": "595.71.05",
            "index": 0,
            "memory_mib": 12288,
            "name": "Fixture GPU",
            "uuid": "GPU-00000000-0000-0000-0000-000000000000",
        }
        self.docker_fingerprint = {
            "daemon_id": "00000000-0000-0000-0000-000000000001",
            "root": str(self.docker_root / "data"),
            "default_runtime": "runc",
            "networks": {"host": "host", "none": "null"},
        }
        self.image_fingerprint = {
            "reference": "fixture-runtime:9.0",
            "id": self.image_id,
            "base_digest": "sha256:" + ("1" * 64),
            "parent_build_image_reference": "fixture-build:9.0",
            "parent_build_image_id": "sha256:" + ("2" * 64),
            "parent_rootfs_layer_count": 2,
            "runtime_rootfs_layer_count": 3,
            "parent_rootfs_sha256": "3" * 64,
            "runtime_rootfs_sha256": "4" * 64,
        }
        self.host_fingerprint = {
            "gpu": copy.deepcopy(self.gpu),
            "docker": copy.deepcopy(self.docker_fingerprint),
            "image": copy.deepcopy(self.image_fingerprint),
            "image_environment": [],
        }
        self.now_utc = datetime(2026, 7, 11, 0, 1, tzinfo=timezone.utc)
        self.launcher = (
            self.runtime_root / "evidence" / self.session_id / "launcher"
        )
        for directory in (
            self.runtime_root / "evidence",
            self.runtime_root / "evidence" / self.session_id,
            self.launcher,
        ):
            directory.mkdir(exist_ok=True)
            directory.chmod(0o700)
        self.write_runtime_evidence()

    @property
    def realization_sha256(self) -> str:
        return _sha256(self.realization_path.read_bytes())

    def write_realization(self) -> None:
        _private_json(self.realization_path, self.realization)

    def asset_evidence(self) -> dict[str, object]:
        return {
            "profiles": ["canonical"],
            "realization_sha256": self.realization_sha256,
            "base_manifest_sha256": self.manifest_sha256,
            "source_contracts_sha256": self.source_contracts_sha256,
            "runtime_image_id": self.image_id,
            "artifact_ids": [self.artifact_id],
            "output_sha256": {self.artifact_id: self.output_sha256},
        }

    def runtime_evidence(self) -> dict[str, object]:
        return {
            **self.asset_evidence(),
            "session_id": self.session_id,
            "lane": "baseline",
            "checksum_manifest": "launcher/SHA256SUMS",
            "checksum_sha256": _sha256(
                (self.launcher / "SHA256SUMS").read_bytes()
            ),
            "behavior_documents": ["fixture_behavior_v1"],
        }

    def matrix(self, evidence_type: str, evidence: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": 2,
            "matrix_id": ownership.EXPECTED_MATRIX_ID,
            "policy": {
                "allowed_module_classifications": sorted(
                    ownership.ALLOWED_MODULE_CLASSIFICATIONS
                ),
                "allowed_capability_statuses": sorted(
                    ownership.ALLOWED_CAPABILITY_STATUSES
                ),
                "strict_blocking_statuses": sorted(
                    ownership.STRICT_BLOCKING_STATUSES
                ),
                "forbidden_ds9_imports": [],
                "scan_roots": ["DS9/noesis", "DS9/scripts"],
                "scan_suffixes": [".py"],
            },
            "modules": [
                {
                    "id": "shared.fixture",
                    "classification": "shared_single_source",
                    "owner_path": "source.py",
                    "rationale": "typed evidence fixture",
                }
            ],
            "capabilities": [
                {
                    "id": "fixture.capability",
                    "surface": "test",
                    "status": "parity",
                    "evidence": {evidence_type: {"ds9": evidence}},
                }
            ],
        }

    def fake_validator(
        self,
        *,
        blocked: bool = False,
        validation_callback: object | None = None,
    ):
        realization_sha256 = self.realization_sha256

        def validate_asset_realization(
            _manifest,
            _realization,
            _artifact_root,
            *,
            profile,
            check_files,
            require_provenance,
        ):
            if callable(validation_callback):
                validation_callback()
            assert check_files is True
            assert require_provenance is True
            return {
                "ok": not blocked,
                "profile": profile,
                "errors": ["fixture profile blocked"] if blocked else [],
                "blockers": ["fixture output absent"] if blocked else [],
                "realization_sha256": realization_sha256,
            }

        return SimpleNamespace(validate_asset_realization=validate_asset_realization)

    def write_runtime_evidence(self) -> None:
        binding = self.asset_evidence()
        profile_result = {
            "ok": True,
            "profile": "canonical",
            "errors": [],
            "blockers": [],
            "realization_sha256": binding["realization_sha256"],
        }
        self.plan = {
            "schema_version": 1,
            "contract": ownership.RUNTIME_CONTRACT,
            "mode": "plan",
            "created_at_utc": "2026-07-10T23:59:50Z",
            "session_id": self.session_id,
            "runtime_lane": "baseline",
            "ready_for_explicit_run": True,
            "blockers": [],
            "unavailable_ports": [],
            "docker_run_invoked": False,
            "owners": {
                "runtime_containers": [],
                "runtime_processes": [],
                "gpu_compute": [],
            },
            "artifact_transaction_lock": {
                "available": True,
                "path": str(
                    self.artifact_root
                    / ".noesis-ds9-artifact-transaction.lock"
                ),
            },
            "host_roots": {
                "artifacts": str(self.artifact_root),
                "runtime": str(self.runtime_root),
            },
            "image": copy.deepcopy(self.image_fingerprint),
            "secondary_docker": copy.deepcopy(self.docker_fingerprint),
            "canonical_runtime": {
                "lane": "baseline",
                "pipeline": "DS9/config/infer.yaml",
                "cameras": "config/cameras.yaml",
                "pgie_profile": "yolo26",
                "model_size": "m",
                "tracking_mode": "baseline",
                "artifact_profiles": ["canonical"],
                "required_engine_ids": [self.artifact_id],
                "source_ids": ["0", "1", "2"],
                "ports": dict(ownership.CANONICAL_PORTS),
                "endpoints": dict(ownership.CANONICAL_ENDPOINTS),
            },
            "session_paths": {
                "build": str(self.runtime_root / "build" / self.session_id),
                "state": str(self.runtime_root / "state" / self.session_id),
                "depth": str(self.runtime_root / "depth" / self.session_id),
                "runtime_evidence": str(
                    self.runtime_root
                    / "evidence"
                    / self.session_id
                    / "runtime"
                ),
                "launcher_evidence": str(self.launcher),
            },
            "artifact_readiness": {
                "ok": True,
                "lane": "baseline",
                "profiles": ["canonical"],
                "realization_sha256": binding["realization_sha256"],
                "base_manifest_sha256": binding["base_manifest_sha256"],
                "engine_source_contracts_sha256": binding[
                    "source_contracts_sha256"
                ],
                "required_engine_ids": [self.artifact_id],
                "host_compatibility": {"host_gpu": copy.deepcopy(self.gpu)},
                "profile_results": {"canonical": profile_result},
            },
            "checkout_snapshot": copy.deepcopy(self.snapshot),
        }
        self.runtime_identity = {
            "schema_version": 1,
            "contract": ownership.RUNTIME_IDENTITY_CONTRACT,
            "contract_version": 1,
            "session_id": self.session_id,
            "runtime_lane": "baseline",
            "runtime_instance_id": "runtime-instance-test",
            "runtime_run_id": "runtime-run-test",
            "health_generated_at_us": 1_000_000,
            "observed_at_utc": "2026-07-11T00:00:01Z",
            "endpoints": dict(ownership.CANONICAL_ENDPOINTS),
        }
        self.summary = {
            "schema_version": 1,
            "contract": ownership.RUNTIME_CONTRACT,
            "mode": "run",
            "session_id": self.session_id,
            "runtime_lane": "baseline",
            "started_at_utc": "2026-07-11T00:00:00Z",
            "finished_at_utc": "2026-07-11T00:00:40.200000Z",
            "ok": True,
            "error": None,
            "container": {
                "name": f"noesis-ds9-runtime-{self.session_id}",
                "started": True,
                "term_sent": True,
                "exit_code": 0,
                "stop_reason": "duration_complete",
                "forced_removal": False,
                "removed": True,
                "absent_after": True,
            },
            "ports_closed_after": True,
            "artifact_transaction_lock": {
                "path": str(
                    self.artifact_root
                    / ".noesis-ds9-artifact-transaction.lock"
                ),
                "held_through_readiness_and_gpu_confirmation": True,
            },
            "gpu_owner_confirmation": {
                "compute_owners": ["1234, python3, 1024"],
                "container_init_pid": 1234,
            },
            "runtime_identity": copy.deepcopy(self.runtime_identity),
            "shutdown_lifecycle": {
                "ok": True,
                "missing_markers": [],
                "ordered": True,
                "failure_signatures": [],
                "severity_signatures": [],
            },
            "checkout": {
                "unchanged": True,
                "before": copy.deepcopy(self.snapshot),
                "after": copy.deepcopy(self.snapshot),
                "added": [],
                "changed": [],
                "removed": [],
                "differences_truncated": False,
            },
            "evidence_directory": str(self.launcher),
        }
        lane = ownership.RUNTIME_LANES["baseline"]
        known_mounts = {
            "/workspace": (self.repo, True),
            "/opt/noesis/ds9-artifacts": (self.artifact_root, True),
            "/var/lib/noesis/build": (
                self.runtime_root / "build" / self.session_id,
                False,
            ),
            "/var/lib/noesis/state": (
                self.runtime_root / "state" / self.session_id,
                False,
            ),
            "/var/lib/noesis/depth": (
                self.runtime_root / "depth" / self.session_id,
                False,
            ),
            "/var/lib/noesis/evidence": (
                self.runtime_root / "evidence" / self.session_id / "runtime",
                False,
            ),
            "/var/lib/noesis/state/analytics": (
                self.runtime_root / "persistent" / "analytics",
                False,
            ),
        }
        for source, read_only in known_mounts.values():
            source.mkdir(parents=True, exist_ok=True)
            if not read_only:
                current = source
                while current != self.runtime_root:
                    current.chmod(0o700)
                    current = current.parent
        secret_root = self.repo.parent / "secrets"
        secret_root.mkdir()
        secret_mounts = {
            "/run/noesis-secrets/camera_sources.json": secret_root
            / "camera_sources.json",
            "/run/noesis-secrets/mapanything_rpc.key": secret_root
            / "mapanything_rpc.key",
            "/run/noesis-secrets/gateway-token": secret_root / "gateway-token",
        }
        for secret in secret_mounts.values():
            _private_write(secret, b"fixture-secret\n")
        all_mounts = {
            **known_mounts,
            **{
                destination: (source, True)
                for destination, source in secret_mounts.items()
            },
        }
        hostconfig_mounts = [
            {
                "Type": "bind",
                "Source": str(source),
                "Target": destination,
                "ReadOnly": read_only,
                "BindOptions": {"Propagation": "rprivate"},
            }
            for destination, (source, read_only) in all_mounts.items()
        ]
        inspect_mounts = [
            {
                "Type": "bind",
                "Source": str(source),
                "Destination": destination,
                "Mode": "",
                "RW": not read_only,
                "Propagation": "rprivate",
            }
            for destination, (source, read_only) in all_mounts.items()
        ]
        self.inspect = [
            {
                "AppArmorProfile": ownership.EXPECTED_APPARMOR_PROFILE,
                "Id": self.container_id,
                "Image": self.image_id,
                "Name": f"/noesis-ds9-runtime-{self.session_id}",
                "State": {
                    "Status": "exited",
                    "Running": False,
                    "ExitCode": 0,
                    "OOMKilled": False,
                    "Error": "",
                    "StartedAt": "2026-07-11T00:00:00.100000000Z",
                    "FinishedAt": "2026-07-11T00:00:40.100000000Z",
                },
                "Config": {
                    "User": f"{os.geteuid()}:{os.getegid()}",
                    "WorkingDir": "/workspace",
                    "Entrypoint": ["python3"],
                    "Cmd": ownership._runtime_arguments(lane),
                    "Env": [
                        f"{key}={value}"
                        for key, value in sorted(
                            ownership._required_runtime_environment(
                                session_id=self.session_id,
                                lane=lane,
                            ).items()
                        )
                    ],
                    "Labels": {
                        "com.noesis.role": "ds9-runtime",
                        "com.noesis.session": self.session_id,
                        "com.noesis.runtime-lane": "baseline",
                    },
                },
                "HostConfig": {
                    "ReadonlyRootfs": True,
                    "Privileged": False,
                    "CapAdd": None,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": list(ownership.EXPECTED_SECURITY_OPTIONS),
                    **copy.deepcopy(dict(ownership.EXPECTED_SENSITIVE_HOST_DEFAULTS)),
                    "Devices": [],
                    "Binds": None,
                    "PidMode": "",
                    "UsernsMode": "",
                    "ReadonlyPaths": list(ownership.EXPECTED_READONLY_PATHS),
                    "MaskedPaths": list(ownership.EXPECTED_MASKED_PATHS),
                    "Memory": 26 * 1024 * 1024 * 1024,
                    "MemorySwap": 26 * 1024 * 1024 * 1024,
                    "PidsLimit": 4096,
                    "Tmpfs": {
                        "/tmp": (
                            "rw,exec,nosuid,nodev,size=2147483648,"
                            f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
                        ),
                        "/run/noesis-secrets": (
                            "rw,noexec,nosuid,nodev,size=65536,"
                            f"uid={os.geteuid()},gid={os.getegid()},mode=0700"
                        ),
                    },
                    "Init": True,
                    "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                    "Ulimits": [{"Name": "nofile", "Hard": 65536, "Soft": 65536}],
                    "LogConfig": {
                        "Type": "json-file",
                        "Config": {"max-file": "2", "max-size": "50m"},
                    },
                    "NetworkMode": "host",
                    "IpcMode": "host",
                    "Runtime": "nvidia",
                    "DeviceRequests": [
                        {
                            "Driver": "",
                            "Count": 0,
                            "DeviceIDs": ["0"],
                            "Capabilities": [["gpu"]],
                            "Options": {},
                        }
                    ],
                    "Mounts": hostconfig_mounts,
                },
                "Mounts": inspect_mounts,
            }
        ]
        self.runtime_log = "\n".join(ownership.SHUTDOWN_MARKERS) + "\n"
        _private_json(self.launcher / "launch-plan.json", self.plan)
        _private_json(self.launcher / "checkout-before.json", self.snapshot)
        _private_json(self.launcher / "summary.json", self.summary)
        _private_json(self.launcher / "container-inspect.json", self.inspect)
        _private_json(self.launcher / "runtime-identity.json", self.runtime_identity)
        _private_json(
            self.launcher / "fixture-behavior.json",
            {
                "schema_version": 1,
                "contract": "noesis.ds9.fixture-behavior",
                "contract_version": 1,
                "session_id": self.session_id,
                "runtime_lane": "baseline",
                "runtime_instance_id": "runtime-instance-test",
                "runtime_run_id": "runtime-run-test",
                "ok": True,
                "frames": 3,
            },
        )
        _private_write(self.launcher / "runtime.log", self.runtime_log.encode())
        self.seal()

    def rewrite_json(self, name: str, payload: object, *, seal: bool = True) -> None:
        _private_json(self.launcher / name, payload)
        if seal:
            self.seal()

    def seal(self) -> None:
        rows = []
        for path in sorted(self.launcher.iterdir()):
            if path.name == "SHA256SUMS":
                continue
            rows.append(f"{_sha256(path.read_bytes())}  {path.name}")
        _private_write(
            self.launcher / "SHA256SUMS", ("\n".join(rows) + "\n").encode()
        )


class TypedRuntimeOwnershipTests(unittest.TestCase):
    def validate(
        self,
        fixture: TypedEvidenceFixture,
        matrix: dict[str, object],
        *,
        blocked: bool = False,
        current_checkout: dict[str, object] | None = None,
        host_fingerprint: dict[str, object] | None = None,
        now_utc: datetime | None = None,
        artifact_root: Path | None = None,
        runtime_root: Path | None = None,
        checkout_side_effect: object | None = None,
        behavior_structure_side_effect: object | None = None,
        asset_validation_callback: object | None = None,
        promoted_evidence: dict[str, object] | None = None,
        promotion_events: dict[str, object] | None = None,
        allow_inline_dynamic_evidence: bool = True,
    ) -> dict[str, object]:
        evidence_types = set(matrix["capabilities"][0]["evidence"])
        external_runtime = bool(
            promoted_evidence
            and promoted_evidence.get("fixture.capability", {})
            .get("runtime_session", {})
        )
        if "runtime_session" in evidence_types or external_runtime:
            registry = {
                "fixture.capability": {
                    "surface": "test",
                    "minimum_tier": "runtime_session",
                    "runtime_requirements": {
                        "baseline": ["fixture_behavior_v1"]
                    },
                }
            }
        else:
            registry = {
                "fixture.capability": {
                    "surface": "test",
                    "minimum_tier": "asset_realization",
                    "required_profiles": ["canonical"],
                }
            }
        behavior_registry = {
            "fixture_behavior_v1": {
                "filename": "fixture-behavior.json",
                "contract": "noesis.ds9.fixture-behavior",
                "lanes": {"baseline"},
                "assertions": (
                    ownership._equals("/ok", True),
                    ownership._minimum("/frames", 1, int),
                ),
            }
        }
        checkout_mock = mock.Mock()
        if checkout_side_effect is None:
            checkout_mock.return_value = current_checkout or fixture.snapshot
        else:
            checkout_mock.side_effect = checkout_side_effect
        behavior_structure_mock = mock.Mock(
            side_effect=behavior_structure_side_effect
        )
        with (
            mock.patch.object(ownership, "REPO_ROOT", fixture.repo),
            mock.patch.object(ownership, "CAPABILITY_REGISTRY", registry),
            mock.patch.object(ownership, "BEHAVIOR_CONTRACTS", behavior_registry),
            mock.patch.object(
                ownership,
                "_validate_behavior_structure",
                new=behavior_structure_mock,
            ),
            mock.patch.object(
                ownership,
                "_load_asset_validator",
                return_value=fixture.fake_validator(
                    blocked=blocked,
                    validation_callback=asset_validation_callback,
                ),
            ),
            mock.patch.object(
                ownership,
                "_current_checkout_summary",
                new=checkout_mock,
            ),
            mock.patch.object(
                ownership,
                "_current_host_fingerprint",
                return_value=host_fingerprint or fixture.host_fingerprint,
            ),
            mock.patch.object(
                ownership,
                "_utc_now",
                return_value=now_utc or fixture.now_utc,
            ),
        ):
            return ownership.validate_matrix(
                matrix,
                artifact_root=artifact_root,
                runtime_root=runtime_root,
                docker_root=fixture.docker_root,
                promoted_evidence=promoted_evidence,
                promotion_events=promotion_events,
                allow_inline_dynamic_evidence=allow_inline_dynamic_evidence,
            )

    def test_asset_realization_binds_selected_outputs_and_authorities(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            matrix = fixture.matrix("asset_realization", fixture.asset_evidence())
            result = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
            )
            self.assertTrue(result["ok"], result["errors"])
            detail = result["evidence_details"]["fixture.capability"][
                "asset_realization"
            ]["ds9"]
            self.assertEqual(detail["artifact_ids"], [fixture.artifact_id])
            self.assertEqual(
                detail["output_sha256"],
                {fixture.artifact_id: fixture.output_sha256},
            )
            self.assertEqual(detail["runtime_image_id"], fixture.image_id)

    def test_asset_selector_builder_closes_over_validated_runtime_image(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            with (
                mock.patch.object(ownership, "REPO_ROOT", fixture.repo),
                mock.patch.object(
                    ownership,
                    "_load_asset_validator",
                    return_value=fixture.fake_validator(),
                ),
            ):
                selector, detail = ownership.build_asset_realization_selector(
                    profiles=["canonical"],
                    artifact_root=fixture.artifact_root,
                )

            self.assertEqual(set(selector), set(ownership.ASSET_REALIZATION_KEYS))
            self.assertEqual(
                selector,
                {key: detail[key] for key in ownership.ASSET_REALIZATION_KEYS},
            )
            self.assertEqual(selector["runtime_image_id"], fixture.image_id)

    def test_asset_selector_runtime_image_is_required_and_authoritative(self) -> None:
        for mutation in ("missing", "malformed", "mismatch"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.asset_evidence()
                if mutation == "missing":
                    del evidence["runtime_image_id"]
                elif mutation == "malformed":
                    evidence["runtime_image_id"] = "not-a-content-addressed-image"
                else:
                    evidence["runtime_image_id"] = "sha256:" + ("f" * 64)

                result = self.validate(
                    fixture,
                    fixture.matrix("asset_realization", evidence),
                    artifact_root=fixture.artifact_root,
                )

                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("runtime_image_id" in error or "runtime image authority drift" in error for error in result["errors"]),
                    result["errors"],
                )

    def test_external_asset_promotion_clears_evidence_blocker_without_inline_selector(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            selector = fixture.asset_evidence()
            matrix = fixture.matrix("asset_realization", selector)
            matrix["capabilities"][0]["evidence"] = {
                "repository_source": {
                    "shared": {"path": "source.py", "contains_all": ["MARKER"]}
                }
            }
            promoted = {
                "fixture.capability": {
                    "asset_realization": {"ds9": selector}
                }
            }
            event = {
                "capability_id": "fixture.capability",
                "evidence_type": "asset_realization",
                "subject": "ds9",
                "selector": selector,
                "checkout_sha256": fixture.snapshot["sha256"],
                "artifact_binding": selector,
                "runtime_binding": None,
                "evidence_sha256s_digest": None,
            }
            events = {
                "fixture.capability": {
                    "asset_realization": {"ds9": event}
                }
            }

            result = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
                current_checkout=fixture.snapshot,
                promoted_evidence=promoted,
                promotion_events=events,
                allow_inline_dynamic_evidence=False,
            )

            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["blockers"], [])
            self.assertEqual(result["evidence_counts"]["asset_realization"], 1)

    def test_normative_matrix_rejects_inline_dynamic_selector(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            result = self.validate(
                fixture,
                fixture.matrix("asset_realization", fixture.asset_evidence()),
                artifact_root=fixture.artifact_root,
                allow_inline_dynamic_evidence=False,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("external registry" in error for error in result["errors"]),
                result["errors"],
            )

    def test_external_runtime_promotion_binds_current_session_and_event(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            selector = fixture.runtime_evidence()
            matrix = fixture.matrix("runtime_session", selector)
            matrix["capabilities"][0]["evidence"] = {
                "repository_source": {
                    "shared": {"path": "source.py", "contains_all": ["MARKER"]}
                }
            }
            promoted = {
                "fixture.capability": {
                    "runtime_session": {"ds9": selector}
                }
            }
            event = {
                "capability_id": "fixture.capability",
                "evidence_type": "runtime_session",
                "subject": "ds9",
                "selector": selector,
                "checkout_sha256": fixture.snapshot["sha256"],
                "artifact_binding": {
                    key: selector[key] for key in ownership.ASSET_REALIZATION_KEYS
                },
                "runtime_binding": {
                    "session_id": fixture.session_id,
                    "lane": "baseline",
                    "runtime_instance_id": "runtime-instance-test",
                    "runtime_run_id": "runtime-run-test",
                },
                "evidence_sha256s_digest": selector["checksum_sha256"],
            }
            events = {
                "fixture.capability": {
                    "runtime_session": {"ds9": event}
                }
            }
            result = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
                promoted_evidence=promoted,
                promotion_events=events,
                allow_inline_dynamic_evidence=False,
            )
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["blockers"], [])

            event["runtime_binding"]["session_id"] = "other-session"
            rejected = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
                promoted_evidence=promoted,
                promotion_events=events,
                allow_inline_dynamic_evidence=False,
            )
            self.assertFalse(rejected["ok"])
            self.assertTrue(
                any("runtime/evidence binding drifted" in error for error in rejected["errors"]),
                rejected["errors"],
            )

    def test_runtime_session_binds_lifecycle_behavior_and_current_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            host = fixture.inspect[0]["HostConfig"]
            self.assertIn("Dns", host)
            self.assertIsNone(host["Dns"])
            self.assertNotIn("StorageOpt", host)
            self.assertNotIn("Sysctls", host)
            matrix = fixture.matrix("runtime_session", fixture.runtime_evidence())
            result = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
            )
            self.assertTrue(result["ok"], result["errors"])
            detail = result["evidence_details"]["fixture.capability"][
                "runtime_session"
            ]["ds9"]
            self.assertEqual(detail["lane"], "baseline")
            self.assertEqual(detail["container_id"], fixture.container_id)
            self.assertEqual(detail["checkout_sha256"], fixture.snapshot["sha256"])

    def test_runtime_plan_binds_exact_session_paths_sources_and_canary_lifecycle(
        self,
    ) -> None:
        mutations = (
            "alternate_build",
            "state",
            "depth",
            "runtime_evidence",
            "extra_session_path",
            "source_subset",
            "source_reordered",
            "source_boolean",
            "appliance_deployment",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                if mutation == "alternate_build":
                    alternate = fixture.runtime_root / "alternate-build"
                    alternate.mkdir(mode=0o700)
                    fixture.plan["session_paths"]["build"] = str(alternate)
                elif mutation in {"state", "depth", "runtime_evidence"}:
                    fixture.plan["session_paths"][mutation] = str(
                        fixture.runtime_root / f"attacker-{mutation}"
                    )
                elif mutation == "extra_session_path":
                    fixture.plan["session_paths"]["unexpected"] = str(
                        fixture.runtime_root / "unexpected"
                    )
                elif mutation == "source_subset":
                    fixture.plan["canonical_runtime"]["source_ids"] = ["0", "1"]
                elif mutation == "source_reordered":
                    fixture.plan["canonical_runtime"]["source_ids"] = ["2", "1", "0"]
                elif mutation == "source_boolean":
                    fixture.plan["canonical_runtime"]["source_ids"] = [False, "1", "2"]
                else:
                    fixture.plan["appliance_deployment"] = {
                        "state_release_id": "state-release-test"
                    }
                fixture.rewrite_json("launch-plan.json", fixture.plan)
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", fixture.runtime_evidence()),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"], mutation)
                expected = (
                    "isolated ephemeral canary"
                    if mutation == "appliance_deployment"
                    else (
                        "source_ids drift"
                        if mutation.startswith("source_")
                        else "session paths are not exact"
                    )
                )
                self.assertTrue(
                    any(expected in error for error in result["errors"]),
                    result["errors"],
                )

    def test_typed_private_evidence_requires_explicit_roots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            matrix = fixture.matrix("runtime_session", fixture.runtime_evidence())
            result = self.validate(fixture, matrix)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("explicit absolute path" in error for error in result["errors"]),
                result["errors"],
            )

    def test_selector_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            evidence = fixture.runtime_evidence()
            evidence["checksum_manifest"] = "launcher/../SHA256SUMS"
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", evidence),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("checksum_manifest" in error for error in result["errors"]))

    def test_private_file_mode_symlink_and_hardlink_are_rejected(self) -> None:
        mutations = ("mode", "symlink", "hardlink")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                if mutation == "mode":
                    (fixture.launcher / "summary.json").chmod(0o644)
                elif mutation == "symlink":
                    behavior = fixture.launcher / "fixture-behavior.json"
                    target = fixture.launcher.parent / "linked-behavior.json"
                    _private_json(target, {"behavior": {"ok": True, "frames": 3}})
                    behavior.unlink()
                    behavior.symlink_to(target)
                else:
                    os.link(
                        fixture.launcher / "summary.json",
                        fixture.launcher.parent / "summary-hardlink.json",
                    )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"], mutation)
                self.assertTrue(
                    any(
                        token in error
                        for error in result["errors"]
                        for token in (
                            "mode-0600",
                            "mode 0600",
                            "symlink",
                            "single-link",
                            "unsafe entry",
                        )
                    ),
                    result["errors"],
                )

    def test_checksum_and_selected_output_drift_are_rejected(self) -> None:
        for mutation in ("checksum", "output"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                if mutation == "checksum":
                    _private_json(
                        fixture.launcher / "fixture-behavior.json",
                        {
                            "schema_version": 1,
                            "contract": "noesis.ds9.fixture-behavior",
                            "session_id": fixture.session_id,
                            "runtime_lane": "baseline",
                            "ok": False,
                            "frames": 3,
                        },
                    )
                else:
                    fixture.output.write_bytes(b"drifted-engine")
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("digest drift" in error or "checksum mismatch" in error for error in result["errors"]),
                    result["errors"],
                )

    def test_checksum_manifest_must_cover_every_launcher_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            evidence = fixture.runtime_evidence()
            _private_json(fixture.launcher / "uncovered.json", {"ok": True})
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", evidence),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("coverage drift" in error for error in result["errors"]))

    def test_manifest_and_source_contract_authority_drift_are_rejected(self) -> None:
        for mutation in ("manifest", "source_contracts"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.asset_evidence()
                if mutation == "manifest":
                    fixture.manifest_path.write_text(
                        fixture.manifest_path.read_text(encoding="utf-8") + "# drift\n",
                        encoding="utf-8",
                    )
                else:
                    fixture.source_contracts_path.write_text(
                        json.dumps(
                            {"schema_version": 1, "contracts": {"new": {}}},
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                result = self.validate(
                    fixture,
                    fixture.matrix("asset_realization", evidence),
                    artifact_root=fixture.artifact_root,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(any("authority" in error or "digest drift" in error for error in result["errors"]))

    def test_realization_drift_and_unrelated_addition_invalidate_session(self) -> None:
        for mutation in ("timestamp", "unrelated_artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                if mutation == "timestamp":
                    fixture.realization["updated_at_utc"] = "2026-07-11T01:00:00Z"
                else:
                    fixture.realization["artifacts"]["engine.unrelated"] = {
                        "state": "staged_unverified",
                        "provenance": {"output_sha256": "d" * 64},
                    }
                fixture.write_realization()
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(any("realization digest drift" in error for error in result["errors"]))

    def test_authoritative_profile_blockers_are_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            matrix = fixture.matrix("asset_realization", fixture.asset_evidence())
            result = self.validate(
                fixture,
                matrix,
                artifact_root=fixture.artifact_root,
                blocked=True,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("authoritative asset profile" in error for error in result["errors"]))
            self.assertTrue(any("fixture output absent" in error for error in result["errors"]))

    def test_incomplete_behavior_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            evidence = fixture.runtime_evidence()
            evidence["behavior_documents"] = []
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", evidence),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("non-empty list" in error for error in result["errors"]))

    def test_historical_checkout_or_native_drift_rejects_behavior_claim(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            current = copy.deepcopy(fixture.snapshot)
            current["sha256"] = "e" * 64
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=current,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(any("historical" in error for error in result["errors"]))

    def test_checkout_change_during_final_validation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            changed = copy.deepcopy(fixture.snapshot)
            changed["sha256"] = "e" * 64
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                checkout_side_effect=[fixture.snapshot, changed],
            )
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("checkout changed during final validation" in error for error in result["errors"]),
                result["errors"],
            )

    def test_artifact_change_during_final_validation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            mutated = False

            def mutate_artifact(*_args: object, **_kwargs: object) -> None:
                nonlocal mutated
                if mutated:
                    return
                mutated = True
                fixture.realization["updated_at_utc"] = "2026-07-11T01:00:00Z"
                fixture.write_realization()

            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
                behavior_structure_side_effect=mutate_artifact,
            )
            self.assertTrue(mutated)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("final CAS" in error for error in result["errors"]),
                result["errors"],
            )

    def test_artifact_change_from_final_checkout_callback_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            checkout_calls = 0
            lock_was_held = False

            def checkout_and_mutate_artifact() -> dict[str, object]:
                nonlocal checkout_calls, lock_was_held
                checkout_calls += 1
                if checkout_calls == 2:
                    lock_path = (
                        fixture.artifact_root
                        / ".noesis-ds9-artifact-transaction.lock"
                    )
                    descriptor = os.open(lock_path, os.O_RDONLY)
                    try:
                        try:
                            fcntl.flock(
                                descriptor,
                                fcntl.LOCK_EX | fcntl.LOCK_NB,
                            )
                        except BlockingIOError:
                            lock_was_held = True
                        else:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)
                    fixture.realization["updated_at_utc"] = (
                        "2026-07-11T01:00:00Z"
                    )
                    fixture.write_realization()
                return fixture.snapshot

            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                checkout_side_effect=checkout_and_mutate_artifact,
            )
            self.assertEqual(checkout_calls, 2)
            self.assertTrue(lock_was_held)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("final CAS after checkout" in error for error in result["errors"]),
                result["errors"],
            )

    def test_launcher_change_during_terminal_artifact_validation_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            validation_calls = 0

            def mutate_launcher_on_terminal_validation() -> None:
                nonlocal validation_calls
                validation_calls += 1
                if validation_calls == 3:
                    _private_write(
                        fixture.launcher / "runtime.log",
                        fixture.runtime_log.encode() + b"terminal mutation\n",
                    )
                    fixture.seal()

            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
                asset_validation_callback=mutate_launcher_on_terminal_validation,
            )
            self.assertEqual(validation_calls, 3)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any(
                    token in error
                    for error in result["errors"]
                    for token in (
                        "launcher evidence changed during terminal artifact validation",
                        "checksum-manifest digest drift",
                    )
                ),
                result["errors"],
            )

    def test_checkout_change_during_terminal_artifact_validation_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            live_checkout = copy.deepcopy(fixture.snapshot)
            validation_calls = 0

            def mutate_checkout_on_terminal_validation() -> None:
                nonlocal validation_calls
                validation_calls += 1
                if validation_calls == 3:
                    live_checkout["sha256"] = "e" * 64

            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", fixture.runtime_evidence()),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=live_checkout,
                asset_validation_callback=mutate_checkout_on_terminal_validation,
            )
            self.assertEqual(validation_calls, 3)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any(
                    "checkout changed during terminal artifact validation" in error
                    for error in result["errors"]
                ),
                result["errors"],
            )

    def test_lane_endpoints_launcher_eos_exit_and_container_identity_are_exact(
        self,
    ) -> None:
        for mutation in (
            "lane",
            "ports",
            "endpoints",
            "launcher",
            "runtime_identity",
            "eos",
            "exit",
            "container",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                if mutation == "lane":
                    evidence["lane"] = "v3dt"
                elif mutation == "ports":
                    fixture.plan["canonical_runtime"]["ports"]["rest"] = 18080
                    fixture.rewrite_json("launch-plan.json", fixture.plan)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "endpoints":
                    fixture.plan["canonical_runtime"]["endpoints"]["rest"] = (
                        "http://127.0.0.1:18080"
                    )
                    fixture.rewrite_json("launch-plan.json", fixture.plan)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "launcher":
                    fixture.plan["session_paths"]["launcher_evidence"] = str(
                        fixture.launcher.with_name("attacker-launcher")
                    )
                    fixture.rewrite_json("launch-plan.json", fixture.plan)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "runtime_identity":
                    fixture.runtime_identity["runtime_instance_id"] = (
                        "attacker-runtime-instance"
                    )
                    fixture.rewrite_json(
                        "runtime-identity.json", fixture.runtime_identity
                    )
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "eos":
                    _private_write(
                        fixture.launcher / "runtime.log",
                        "\n".join(ownership.SHUTDOWN_MARKERS[:-1]).encode() + b"\n",
                    )
                    fixture.seal()
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "exit":
                    fixture.summary["container"]["exit_code"] = 1
                    fixture.rewrite_json("summary.json", fixture.summary)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                else:
                    fixture.inspect[0]["Id"] = "short"
                    fixture.rewrite_json("container-inspect.json", fixture.inspect)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"], mutation)

    def test_attacker_mount_redirect_and_forbidden_env_are_rejected(self) -> None:
        for mutation in (
            "workspace",
            "artifact",
            "forbidden_env",
            "extra_env",
            "extra_mount",
            "duplicate_destination",
            "duplicate_source",
            "secret_overlap",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                inspect = fixture.inspect[0]
                if mutation in {"workspace", "artifact"}:
                    destination = (
                        "/workspace"
                        if mutation == "workspace"
                        else "/opt/noesis/ds9-artifacts"
                    )
                    attacker = fixture.repo.parent / f"attacker-{mutation}"
                    attacker.mkdir()
                    for row in inspect["HostConfig"]["Mounts"]:
                        if row["Target"] == destination:
                            row["Source"] = str(attacker)
                    for row in inspect["Mounts"]:
                        if row["Destination"] == destination:
                            row["Source"] = str(attacker)
                elif mutation == "forbidden_env":
                    inspect["Config"]["Env"].append(
                        "NOESIS_DS9_ALLOW_PYDS_COMPAT=1"
                    )
                elif mutation == "extra_env":
                    inspect["Config"]["Env"].append("PYTHONPATH=/attacker")
                elif mutation == "extra_mount":
                    attacker = fixture.repo.parent / "attacker-extra"
                    attacker.mkdir()
                    inspect["HostConfig"]["Mounts"].append(
                        {
                            "Type": "bind",
                            "Source": str(attacker),
                            "Target": "/attacker",
                            "ReadOnly": False,
                            "BindOptions": {"Propagation": "rprivate"},
                        }
                    )
                    inspect["Mounts"].append(
                        {
                            "Type": "bind",
                            "Source": str(attacker),
                            "Destination": "/attacker",
                            "Mode": "",
                            "RW": True,
                            "Propagation": "rprivate",
                        }
                    )
                elif mutation == "duplicate_destination":
                    inspect["HostConfig"]["Mounts"].append(
                        copy.deepcopy(inspect["HostConfig"]["Mounts"][0])
                    )
                    inspect["Mounts"].append(
                        copy.deepcopy(inspect["Mounts"][0])
                    )
                elif mutation == "duplicate_source":
                    host_row = copy.deepcopy(inspect["HostConfig"]["Mounts"][0])
                    host_row["Target"] = "/duplicate-source"
                    inspect["HostConfig"]["Mounts"].append(host_row)
                    inspect_row = copy.deepcopy(inspect["Mounts"][0])
                    inspect_row["Destination"] = "/duplicate-source"
                    inspect["Mounts"].append(inspect_row)
                else:
                    secret_rows = [
                        row
                        for row in inspect["HostConfig"]["Mounts"]
                        if str(row["Target"]).startswith("/run/noesis-secrets/")
                    ]
                    secret_rows[1]["Source"] = secret_rows[0]["Source"]
                    inspect_secret_rows = [
                        row
                        for row in inspect["Mounts"]
                        if str(row["Destination"]).startswith(
                            "/run/noesis-secrets/"
                        )
                    ]
                    inspect_secret_rows[1]["Source"] = inspect_secret_rows[0][
                        "Source"
                    ]
                fixture.rewrite_json("container-inspect.json", fixture.inspect)
                evidence["checksum_sha256"] = _sha256(
                    (fixture.launcher / "SHA256SUMS").read_bytes()
                )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any(
                        token in error
                        for error in result["errors"]
                        for token in (
                            "bind mount drift",
                            "forbidden override",
                            "exact runtime-image",
                            "extra bind",
                            "unsafe/duplicate bind",
                        )
                    ),
                    result["errors"],
                )

    def test_tmpfs_init_restart_ulimit_and_log_policy_are_exact(self) -> None:
        mutations = {
            "tmpfs": lambda host: host["Tmpfs"].update({"/tmp": "rw,mode=0777"}),
            "init": lambda host: host.__setitem__("Init", False),
            "restart": lambda host: host.__setitem__(
                "RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}
            ),
            "ulimit": lambda host: host.__setitem__("Ulimits", []),
            "log": lambda host: host.__setitem__(
                "LogConfig", {"Type": "json-file", "Config": {}}
            ),
            "restart_bool": lambda host: host.__setitem__(
                "RestartPolicy", {"Name": "no", "MaximumRetryCount": False}
            ),
            "security_extra": lambda host: host.__setitem__(
                "SecurityOpt",
                [*ownership.EXPECTED_SECURITY_OPTIONS, "seccomp=unconfined"],
            ),
            "security_missing": lambda host: host.__setitem__(
                "SecurityOpt", ["no-new-privileges"]
            ),
            "devices": lambda host: host.__setitem__(
                "Devices", [{"PathOnHost": "/dev/mem"}]
            ),
            "binds": lambda host: host.__setitem__("Binds", []),
            "pid_mode": lambda host: host.__setitem__("PidMode", "host"),
            "userns_mode": lambda host: host.__setitem__("UsernsMode", "host"),
            "readonly_paths": lambda host: host.__setitem__("ReadonlyPaths", []),
            "masked_paths": lambda host: host.__setitem__("MaskedPaths", []),
            "cgroupns": lambda host: host.__setitem__("CgroupnsMode", "host"),
            "oom_disable": lambda host: host.__setitem__("OomKillDisable", True),
            "sysctls_null": lambda host: host.__setitem__("Sysctls", None),
            "sysctls": lambda host: host.__setitem__("Sysctls", {}),
            "sysctls_nonempty": lambda host: host.__setitem__(
                "Sysctls", {"net.ipv4.ip_forward": "1"}
            ),
            "memory_swappiness": lambda host: host.__setitem__(
                "MemorySwappiness", 60
            ),
            "device_cgroup_rules": lambda host: host.__setitem__(
                "DeviceCgroupRules", ["c 1:3 rwm"]
            ),
            "cgroup_parent": lambda host: host.__setitem__(
                "CgroupParent", "system.slice"
            ),
            "dns_missing": lambda host: host.pop("Dns"),
            "dns_empty": lambda host: host.__setitem__("Dns", []),
            "dns": lambda host: host.__setitem__("Dns", ["8.8.8.8"]),
            "dns_options": lambda host: host.__setitem__(
                "DnsOptions", ["use-vc"]
            ),
            "dns_search": lambda host: host.__setitem__(
                "DnsSearch", ["example.test"]
            ),
            "storage_opt_null": lambda host: host.__setitem__("StorageOpt", None),
            "storage_opt_empty": lambda host: host.__setitem__("StorageOpt", {}),
            "storage_opt": lambda host: host.__setitem__(
                "StorageOpt", {"size": "20G"}
            ),
            "container_id_file": lambda host: host.__setitem__(
                "ContainerIDFile", "/tmp/container.id"
            ),
            "volume_driver": lambda host: host.__setitem__(
                "VolumeDriver", "local"
            ),
            "oom_score_adj": lambda host: host.__setitem__("OomScoreAdj", 1),
            "isolation": lambda host: host.__setitem__("Isolation", "hyperv"),
            "cgroup": lambda host: host.__setitem__("Cgroup", "custom"),
            "shm_size": lambda host: host.__setitem__(
                "ShmSize", 128 * 1024 * 1024
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                mutate(fixture.inspect[0]["HostConfig"])
                fixture.rewrite_json("container-inspect.json", fixture.inspect)
                evidence["checksum_sha256"] = _sha256(
                    (fixture.launcher / "SHA256SUMS").read_bytes()
                )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(
                    any("container inspection identity" in error for error in result["errors"])
                )

    def test_apparmor_profile_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            evidence = fixture.runtime_evidence()
            fixture.inspect[0]["AppArmorProfile"] = ""
            fixture.rewrite_json("container-inspect.json", fixture.inspect)
            evidence["checksum_sha256"] = _sha256(
                (fixture.launcher / "SHA256SUMS").read_bytes()
            )
            result = self.validate(
                fixture,
                fixture.matrix("runtime_session", evidence),
                artifact_root=fixture.artifact_root,
                runtime_root=fixture.runtime_root,
                current_checkout=fixture.snapshot,
            )
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("container inspection identity" in error for error in result["errors"]),
                result["errors"],
            )

    def test_runtime_integer_fields_reject_boolean_substitution(self) -> None:
        for mutation in ("summary_schema", "summary_exit", "inspect_exit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                if mutation == "summary_schema":
                    fixture.summary["schema_version"] = True
                    fixture.rewrite_json("summary.json", fixture.summary)
                elif mutation == "summary_exit":
                    fixture.summary["container"]["exit_code"] = False
                    fixture.rewrite_json("summary.json", fixture.summary)
                else:
                    fixture.inspect[0]["State"]["ExitCode"] = False
                    fixture.rewrite_json("container-inspect.json", fixture.inspect)
                evidence["checksum_sha256"] = _sha256(
                    (fixture.launcher / "SHA256SUMS").read_bytes()
                )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])

    def test_generic_or_misbound_behavior_document_cannot_prove_capability(self) -> None:
        mutations = (
            "generic",
            "contract",
            "contract_version",
            "schema",
            "session",
            "lane",
            "runtime_instance",
            "runtime_run",
            "value_type",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                behavior_path = fixture.launcher / "fixture-behavior.json"
                behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
                if mutation == "generic":
                    evidence["behavior_documents"] = ["generic_json_ok"]
                elif mutation == "contract":
                    behavior["contract"] = "attacker.generic"
                elif mutation == "contract_version":
                    behavior["contract_version"] = True
                elif mutation == "schema":
                    behavior["schema_version"] = True
                elif mutation == "session":
                    behavior["session_id"] = "other-session"
                elif mutation == "lane":
                    behavior["runtime_lane"] = "v3dt"
                elif mutation == "runtime_instance":
                    behavior["runtime_instance_id"] = "other-runtime-instance"
                elif mutation == "runtime_run":
                    behavior["runtime_run_id"] = "other-runtime-run"
                else:
                    behavior["frames"] = True
                if mutation != "generic":
                    fixture.rewrite_json("fixture-behavior.json", behavior)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
                self.assertFalse(result["ok"])
                self.assertTrue(any("behavior" in error for error in result["errors"]))

    def test_freshness_host_fingerprint_rfc3339_and_duration_gate_live_readiness(self) -> None:
        for mutation in (
            "archive",
            "host_gpu",
            "host_docker",
            "host_image",
            "timestamp",
            "duration",
            "plan_start_slack",
            "summary_start_slack",
            "summary_finish_slack",
            "forged_fresh_summary",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                fixture = TypedEvidenceFixture(Path(raw))
                evidence = fixture.runtime_evidence()
                current_host = copy.deepcopy(fixture.host_fingerprint)
                now = fixture.now_utc
                if mutation == "archive":
                    now = datetime(2026, 7, 13, tzinfo=timezone.utc)
                elif mutation == "host_gpu":
                    current_host["gpu"]["driver_version"] = "999.0"
                elif mutation == "host_docker":
                    current_host["docker"]["daemon_id"] = (
                        "ffffffff-ffff-ffff-ffff-ffffffffffff"
                    )
                elif mutation == "host_image":
                    current_host["image"]["runtime_rootfs_sha256"] = "f" * 64
                elif mutation == "timestamp":
                    fixture.summary["finished_at_utc"] = "2026-07-11 00:00:40"
                    fixture.rewrite_json("summary.json", fixture.summary)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "duration":
                    fixture.inspect[0]["State"]["FinishedAt"] = (
                        "2026-07-11T00:00:05.100000000Z"
                    )
                    fixture.summary["finished_at_utc"] = "2026-07-11T00:00:05.200000Z"
                    fixture.rewrite_json("summary.json", fixture.summary, seal=False)
                    fixture.rewrite_json("container-inspect.json", fixture.inspect)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "plan_start_slack":
                    fixture.plan["created_at_utc"] = "2026-07-10T23:57:00Z"
                    fixture.rewrite_json("launch-plan.json", fixture.plan)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "summary_start_slack":
                    fixture.plan["created_at_utc"] = "2026-07-10T23:58:50Z"
                    fixture.summary["started_at_utc"] = "2026-07-10T23:59:00Z"
                    fixture.rewrite_json("launch-plan.json", fixture.plan, seal=False)
                    fixture.rewrite_json("summary.json", fixture.summary)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                elif mutation == "summary_finish_slack":
                    fixture.summary["finished_at_utc"] = "2026-07-11T00:01:00Z"
                    fixture.rewrite_json("summary.json", fixture.summary)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                else:
                    fixture.plan["created_at_utc"] = "2026-07-09T23:59:50Z"
                    fixture.summary["started_at_utc"] = "2026-07-10T00:00:00Z"
                    fixture.inspect[0]["State"]["StartedAt"] = (
                        "2026-07-10T00:00:00.100000000Z"
                    )
                    fixture.inspect[0]["State"]["FinishedAt"] = (
                        "2026-07-10T00:00:40.100000000Z"
                    )
                    fixture.summary["finished_at_utc"] = "2026-07-11T00:00:40Z"
                    fixture.rewrite_json("launch-plan.json", fixture.plan, seal=False)
                    fixture.rewrite_json("container-inspect.json", fixture.inspect, seal=False)
                    fixture.rewrite_json("summary.json", fixture.summary)
                    evidence["checksum_sha256"] = _sha256(
                        (fixture.launcher / "SHA256SUMS").read_bytes()
                    )
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                    host_fingerprint=current_host,
                    now_utc=now,
                )
                self.assertFalse(result["ok"])
                if mutation in {
                    "archive",
                    "host_gpu",
                    "host_docker",
                    "host_image",
                }:
                    detail = result["evidence_details"]["fixture.capability"][
                        "runtime_session"
                    ]["ds9"]
                    self.assertFalse(detail["live_ready"])
                    self.assertTrue(detail["live_readiness_blockers"])

    def test_descriptor_anchored_parent_rename_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            fixture = TypedEvidenceFixture(Path(raw))
            evidence = fixture.runtime_evidence()
            swapped = False

            def rename_swap(label: str, _root: Path, _relative: Path) -> None:
                nonlocal swapped
                if label != "runtime evidence summary.json" or swapped:
                    return
                swapped = True
                old = fixture.launcher.with_name("launcher-opened")
                fixture.launcher.rename(old)
                fixture.launcher.mkdir(mode=0o700)
                fixture.launcher.chmod(0o700)

            with mock.patch.object(ownership, "_OPENAT_TEST_HOOK", rename_swap):
                result = self.validate(
                    fixture,
                    fixture.matrix("runtime_session", evidence),
                    artifact_root=fixture.artifact_root,
                    runtime_root=fixture.runtime_root,
                    current_checkout=fixture.snapshot,
                )
            self.assertTrue(swapped)
            self.assertFalse(result["ok"])
            self.assertTrue(
                any("replaced" in error or "missing" in error for error in result["errors"]),
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()
