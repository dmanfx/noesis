#!/usr/bin/env python3
"""Promote validated DS9 artifact/session evidence into the private registry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = Path(__file__).resolve().with_name("validate_runtime_ownership.py")
REGISTRY_PATH = Path(__file__).resolve().with_name("runtime_ownership_registry.py")


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load DS9 authority module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "DS9" / "docs" / "runtime_ownership.yaml",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--docker-root", type=Path)
    parser.add_argument("--capability-id", required=True)
    parser.add_argument(
        "--evidence-type",
        choices=("asset_realization", "runtime_session"),
        required=True,
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--lane", choices=("baseline", "v3dt", "wholebody49-s", "wholebody49-x"))
    parser.add_argument("--session-id")
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--supersedes-event-digest")
    parser.add_argument("--revoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _nested_promotions(
    validator: ModuleType,
    snapshot: Any,
    *,
    matrix_id: str,
    matrix_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return validator._active_registry_promotions(
        snapshot,
        matrix_id=matrix_id,
        matrix_sha256=matrix_sha256,
    )


def _put_nested(
    target: dict[str, Any],
    capability_id: str,
    evidence_type: str,
    subject: str,
    value: Mapping[str, Any],
) -> None:
    target.setdefault(capability_id, {}).setdefault(evidence_type, {})[
        subject
    ] = dict(value)


def _artifact_binding(
    validator: ModuleType, detail: Mapping[str, Any]
) -> dict[str, Any]:
    return {key: detail[key] for key in validator.ASSET_REALIZATION_KEYS}


def _selector_for_args(
    validator: ModuleType,
    args: argparse.Namespace,
    matrix: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.artifact_root is None:
        raise ValueError("promotion requires --artifact-root")
    if args.evidence_type == "asset_realization":
        profiles = list(args.profile)
        if not profiles:
            policy = validator.CAPABILITY_REGISTRY[args.capability_id]
            profiles = list(policy.get("required_profiles", []) or [])
        if not profiles:
            raise ValueError(
                "asset promotion requires --profile when the capability has no exact profile policy"
            )
        return validator.build_asset_realization_selector(
            profiles=profiles,
            artifact_root=args.artifact_root,
        )
    if args.lane is None or args.session_id is None or args.docker_root is None:
        raise ValueError(
            "runtime promotion requires --lane, --session-id, and --docker-root"
        )
    return validator.build_runtime_session_selector(
        capability_id=args.capability_id,
        lane=args.lane,
        session_id=args.session_id,
        artifact_root=args.artifact_root,
        runtime_root=args.runtime_root,
        docker_root=args.docker_root,
    )


def promote(args: argparse.Namespace) -> Mapping[str, Any]:
    validator = _load_module(VALIDATOR_PATH, "_noesis_ds9_promotion_validator")
    registry = _load_module(REGISTRY_PATH, "_noesis_ds9_promotion_registry")
    supervisor: ModuleType | None = None
    artifact_lock_descriptor: int | None = None
    try:
        matrix, matrix_raw = validator._load_yaml_with_raw(args.matrix.absolute())
        matrix_id = str(matrix.get("matrix_id") or "")
        matrix_sha256 = validator._sha256_bytes(matrix_raw)
        if matrix_id != validator.EXPECTED_MATRIX_ID:
            raise ValueError("promotion matrix identity drifted")
        capability_rows = {
            row.get("id"): row
            for row in matrix.get("capabilities", [])
            if isinstance(row, Mapping)
        }
        row = capability_rows.get(args.capability_id)
        if not isinstance(row, Mapping) or args.capability_id not in validator.CAPABILITY_REGISTRY:
            raise ValueError("promotion capability is not in the validator registry")
        if row.get("status") in validator.STRICT_BLOCKING_STATUSES:
            raise ValueError(
                "dynamic evidence cannot override a tracked known_gap/blocked capability"
            )
        if registry.SUBJECT_RE.fullmatch(str(args.subject)) is None:
            raise ValueError("promotion subject is unsafe")
        selector: dict[str, Any] | None = None
        detail: dict[str, Any] | None = None
        if not args.revoke:
            selector, detail = _selector_for_args(validator, args, matrix)

        held_artifact_root = (
            args.artifact_root.expanduser().absolute()
            if args.artifact_root is not None
            else None
        )

        def build(snapshot: Any) -> Mapping[str, Any]:
            nonlocal artifact_lock_descriptor, supervisor
            _current_matrix, current_matrix_raw = validator._load_yaml_with_raw(
                args.matrix.absolute()
            )
            if validator._sha256_bytes(current_matrix_raw) != matrix_sha256:
                raise ValueError("promotion matrix changed before registry append")
            key = (args.capability_id, args.evidence_type, args.subject)
            prior = snapshot.key_heads.get(key)
            prior_digest = prior.get("event_digest") if prior is not None else None
            if args.supersedes_event_digest != prior_digest:
                raise ValueError(
                    "--supersedes-event-digest must exactly name the current key head"
                )
            if args.revoke and key not in snapshot.active:
                raise ValueError("revoke requires a currently active promotion")
            checkout = dict(validator._current_checkout_summary())
            checkout_sha256 = validator._require_sha256(
                checkout.get("sha256"), "promotion checkout sha256"
            )
            runtime_binding = None
            evidence_digest = None
            artifact_binding = None
            if detail is not None:
                artifact_binding = _artifact_binding(validator, detail)
                if args.evidence_type == "runtime_session":
                    runtime_binding = {
                        "session_id": detail["session_id"],
                        "lane": detail["lane"],
                        "runtime_instance_id": detail["runtime_instance_id"],
                        "runtime_run_id": detail["runtime_run_id"],
                    }
                    evidence_digest = detail["checksum_sha256"]
            event = registry.build_event(
                snapshot,
                event_type="revoke" if args.revoke else "promote",
                matrix_id=matrix_id,
                matrix_sha256=matrix_sha256,
                capability_id=args.capability_id,
                evidence_type=args.evidence_type,
                subject=args.subject,
                selector=selector,
                runtime_binding=runtime_binding,
                checkout_sha256=checkout_sha256,
                artifact_binding=artifact_binding,
                evidence_sha256s_digest=evidence_digest,
                supersedes_event_digest=args.supersedes_event_digest,
            )
            candidate = registry.validate_event_chain([*snapshot.events, event])
            promoted, events = _nested_promotions(
                validator,
                candidate,
                matrix_id=matrix_id,
                matrix_sha256=matrix_sha256,
            )
            result = validator.validate_matrix(
                matrix,
                artifact_root=args.artifact_root,
                runtime_root=args.runtime_root,
                docker_root=args.docker_root,
                promoted_evidence=promoted,
                promotion_events=events,
            )
            if not result.get("ok"):
                raise ValueError(
                    "candidate promotion failed independent matrix validation: "
                    + "; ".join(str(value) for value in result.get("errors", []))
                )
            if detail is not None:
                validated_detail = (
                    result.get("evidence_details", {})
                    .get(args.capability_id, {})
                    .get(args.evidence_type, {})
                    .get(args.subject)
                )
                if not isinstance(validated_detail, Mapping):
                    raise ValueError("candidate promotion produced no validated detail")
                supervisor = validator._load_runtime_supervisor_module()
                artifact_lock_descriptor, _lock_path = (
                    supervisor.acquire_artifact_transaction_lock(held_artifact_root)
                )
                terminal_artifact = validator._validate_asset_realization_evidence(
                    {key: selector[key] for key in validator.ASSET_REALIZATION_KEYS},
                    artifact_root=held_artifact_root,
                    label="promotion terminal artifact CAS",
                )
                if not validator._strict_json_equal(
                    terminal_artifact, artifact_binding
                ):
                    raise ValueError("promotion artifact changed before registry append")
                terminal_checkout = dict(validator._current_checkout_summary())
                if terminal_checkout.get("sha256") != checkout_sha256:
                    raise ValueError("promotion checkout changed before registry append")
                _terminal_matrix, terminal_matrix_raw = (
                    validator._load_yaml_with_raw(args.matrix.absolute())
                )
                if validator._sha256_bytes(terminal_matrix_raw) != matrix_sha256:
                    raise ValueError("promotion matrix changed before registry append")
                if args.evidence_type == "runtime_session":
                    runtime = validator._external_root(
                        args.runtime_root,
                        "runtime root",
                        accepted_modes=frozenset({0o700}),
                    )
                    launcher_relative = (
                        Path("evidence")
                        / str(selector["session_id"])
                        / "launcher"
                    )
                    checksum_raw = validator._anchored_file_content(
                        runtime,
                        launcher_relative / "SHA256SUMS",
                        "promotion terminal runtime evidence CAS",
                        directory_policy="private",
                        file_policy="private",
                    )
                    if (
                        not isinstance(checksum_raw, bytes)
                        or validator._sha256_bytes(checksum_raw)
                        != selector["checksum_sha256"]
                    ):
                        raise ValueError(
                            "promotion runtime evidence changed before registry append"
                        )
                    validator._load_checksum_covered_files(
                        runtime,
                        launcher_relative,
                        expected_manifest_sha256=selector["checksum_sha256"],
                    )
            return event

        event = registry.append_registry_event(args.runtime_root, build)
        return {
            "ok": True,
            "registry_directory": str(
                args.runtime_root.expanduser().absolute()
                / registry.REGISTRY_DIRECTORY
            ),
            "event": dict(event),
        }
    finally:
        if supervisor is not None:
            try:
                supervisor.release_artifact_transaction_lock(
                    artifact_lock_descriptor
                )
            finally:
                sys.modules.pop(supervisor.__name__, None)
        sys.modules.pop(registry.__name__, None)
        sys.modules.pop(validator.__name__, None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = promote(args)
    except Exception as exc:
        result = {
            "ok": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    if args.json or not result.get("ok"):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        event = result["event"]
        print(
            "[OK] recorded DS9 ownership promotion "
            f"sequence={event['sequence']} digest={event['event_digest']}"
        )
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
