#!/usr/bin/env python3
"""Validate the canonical native DS9.1 ownership and retirement boundary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = REPO_ROOT / "DS9/docs/runtime_ownership.yaml"
EXPECTED_MATRIX_ID = "noesis-ds9-native-runtime-ownership"
EXPECTED_CANONICAL_RUNTIME = "ds9_native_host"
ALLOWED_MODULE_CLASSIFICATIONS = frozenset({"shared_single_source", "ds9_adapter"})
ALLOWED_CAPABILITY_STATUSES = frozenset({"enabled", "opt_in", "deferred"})
REQUIRED_POLICY_PATHS = {
    "entrypoint": "DS9/noesis/ds9_runtime.py",
    "implementation": "DS9/noesis/ds9_runtime_core.py",
    "supervisor": "DS9/scripts/run_canonical_runtime_host.py",
    "pipeline_config": "DS9/config/infer.yaml",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_matrix(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read ownership matrix: {exc}") from exc
    try:
        matrix = yaml.load(payload, Loader=_UniqueKeyLoader)
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"ownership matrix is not strict YAML: {exc}") from exc
    if not isinstance(matrix, Mapping):
        raise ValueError("ownership matrix root must be a mapping")
    return matrix


def _repo_path(raw: object, label: str) -> Path:
    text = str(raw or "")
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a safe repository-relative path")
    return REPO_ROOT / candidate


def _scan_forbidden_references(policy: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    roots = policy.get("scan_roots")
    suffixes = policy.get("scan_suffixes")
    rules = policy.get("forbidden_active_references")
    if not isinstance(roots, list) or not roots:
        return ["policy.scan_roots must be a non-empty list"]
    if not isinstance(suffixes, list) or not suffixes:
        return ["policy.scan_suffixes must be a non-empty list"]
    if not isinstance(rules, list) or not rules:
        return ["policy.forbidden_active_references must be a non-empty list"]

    compiled: list[tuple[str, re.Pattern[str], frozenset[str]]] = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            errors.append(f"forbidden rule {index} must be a mapping")
            continue
        rule_id = str(rule.get("id") or f"rule-{index}")
        raw_allow_paths = rule.get("allow_paths", [])
        if not isinstance(raw_allow_paths, list):
            errors.append(f"forbidden rule {rule_id}.allow_paths must be a list")
            raw_allow_paths = []
        allow_paths: set[str] = set()
        for raw_path in raw_allow_paths:
            try:
                allowed = _repo_path(raw_path, f"forbidden rule {rule_id}.allow_paths")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            allow_paths.add(allowed.relative_to(REPO_ROOT).as_posix())
        try:
            compiled.append(
                (
                    rule_id,
                    re.compile(str(rule.get("regex") or "")),
                    frozenset(allow_paths),
                )
            )
        except re.error as exc:
            errors.append(f"forbidden rule {rule_id} has invalid regex: {exc}")

    allowed_suffixes = {str(item) for item in suffixes}
    validator = Path(__file__).resolve()
    for raw_root in roots:
        try:
            root = _repo_path(raw_root, "scan root")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not root.exists():
            continue
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in allowed_suffixes:
                continue
            if path.resolve() == validator:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"cannot scan {path.relative_to(REPO_ROOT)}: {exc}")
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            for rule_id, pattern, allow_paths in compiled:
                if relative in allow_paths:
                    continue
                match = pattern.search(text)
                if match:
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(f"{relative}:{line}: forbidden {rule_id} reference")
    return errors


def validate_matrix(matrix: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if matrix.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if matrix.get("matrix_id") != EXPECTED_MATRIX_ID:
        errors.append(f"matrix_id must be {EXPECTED_MATRIX_ID!r}")

    policy = matrix.get("policy")
    if not isinstance(policy, Mapping):
        errors.append("policy must be a mapping")
        policy = {}
    if policy.get("canonical_runtime") != EXPECTED_CANONICAL_RUNTIME:
        errors.append(f"policy.canonical_runtime must be {EXPECTED_CANONICAL_RUNTIME!r}")
    for key, expected in REQUIRED_POLICY_PATHS.items():
        if policy.get(key) != expected:
            errors.append(f"policy.{key} must be {expected!r}")
        elif not (REPO_ROOT / expected).is_file():
            errors.append(f"policy.{key} is missing: {expected}")
    if set(policy.get("allowed_module_classifications") or ()) != set(
        ALLOWED_MODULE_CLASSIFICATIONS
    ):
        errors.append("allowed module classifications drifted")

    modules = matrix.get("modules")
    if not isinstance(modules, list) or not modules:
        errors.append("modules must be a non-empty list")
        modules = []
    module_ids: set[str] = set()
    owner_paths: set[str] = set()
    for index, row in enumerate(modules):
        label = f"modules[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        module_id = str(row.get("id") or "")
        if not module_id or module_id in module_ids:
            errors.append(f"{label}.id must be non-empty and unique")
        module_ids.add(module_id)
        classification = str(row.get("classification") or "")
        if classification not in ALLOWED_MODULE_CLASSIFICATIONS:
            errors.append(f"{label}.classification is invalid")
        raw_owner = str(row.get("owner_path") or "")
        if not raw_owner or raw_owner in owner_paths:
            errors.append(f"{label}.owner_path must be non-empty and unique")
        owner_paths.add(raw_owner)
        try:
            owner = _repo_path(raw_owner, f"{label}.owner_path")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not owner.exists():
            errors.append(f"{label}.owner_path is missing: {raw_owner}")
        if classification == "ds9_adapter" and not raw_owner.startswith("DS9/"):
            errors.append(f"{label}: DS9 adapters must be owned under DS9/")
        if classification == "shared_single_source" and raw_owner.startswith("DS9/"):
            errors.append(f"{label}: shared owners must be outside DS9/")

    capabilities = matrix.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty list")
        capabilities = []
    capability_ids: set[str] = set()
    for index, row in enumerate(capabilities):
        label = f"capabilities[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        capability_id = str(row.get("id") or "")
        if not capability_id or capability_id in capability_ids:
            errors.append(f"{label}.id must be non-empty and unique")
        capability_ids.add(capability_id)
        if row.get("status") not in ALLOWED_CAPABILITY_STATUSES:
            errors.append(f"{label}.status is invalid")
        try:
            owner = _repo_path(row.get("owner_path"), f"{label}.owner_path")
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if not owner.exists():
                errors.append(f"{label}.owner_path is missing")

    errors.extend(_scan_forbidden_references(policy))
    return {
        "ok": not errors,
        "matrix_id": matrix.get("matrix_id"),
        "canonical_runtime": policy.get("canonical_runtime"),
        "module_count": len(modules),
        "capability_count": len(capabilities),
        "errors": errors,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        matrix = _load_matrix(args.matrix)
        result = validate_matrix(matrix)
    except ValueError as exc:
        result = {"ok": False, "errors": [str(exc)]}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(
            "[OK] native DS9 ownership validated "
            f"({result['module_count']} modules, {result['capability_count']} capabilities)"
        )
    else:
        for error in result.get("errors", []):
            print(f"[FAIL] {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
