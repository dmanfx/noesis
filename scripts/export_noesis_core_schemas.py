#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noesis_core.contracts import CONTRACT_MODELS


DEFAULT_OUTPUT = REPO_ROOT / "contracts" / "schema"
DEFAULT_TYPESCRIPT_OUTPUT = REPO_ROOT / "contracts" / "typescript" / "noesis-contracts.ts"


def _serialized_schema(model: Any) -> str:
    payload = model.model_json_schema(mode="serialization")
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def export_schemas(output_dir: Path, *, check: bool) -> list[str]:
    output_dir = output_dir.resolve()
    mismatches: list[str] = []
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in sorted(CONTRACT_MODELS.items()):
        path = output_dir / f"{name}.schema.json"
        expected = _serialized_schema(model)
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(str(path))
            continue
        path.write_text(expected, encoding="utf-8")
    return mismatches


def _typescript_property_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _typescript_literal(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False)
    return "unknown"


def _typescript_type(schema: Mapping[str, Any], *, indent: str = "") -> str:
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    if "const" in schema:
        return _typescript_literal(schema["const"])
    if "enum" in schema:
        values = [_typescript_literal(value) for value in schema.get("enum", [])]
        return " | ".join(values) if values else "never"
    if "anyOf" in schema:
        values = []
        for item in schema.get("anyOf", []):
            rendered = _typescript_type(item, indent=indent)
            if rendered not in values:
                values.append(rendered)
        return " | ".join(values) if values else "unknown"
    if "oneOf" in schema:
        values = [_typescript_type(item, indent=indent) for item in schema.get("oneOf", [])]
        return " | ".join(dict.fromkeys(values)) if values else "unknown"
    if "allOf" in schema:
        values = [_typescript_type(item, indent=indent) for item in schema.get("allOf", [])]
        return " & ".join(dict.fromkeys(values)) if values else "unknown"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return " | ".join(
            dict.fromkeys(_typescript_type({**schema, "type": item}, indent=indent) for item in schema_type)
        )
    if schema_type == "null":
        return "null"
    if schema_type == "boolean":
        return "boolean"
    if schema_type in {"integer", "number"}:
        return "number"
    if schema_type == "string":
        return "string"
    if schema_type == "array":
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            return "readonly [" + ", ".join(_typescript_type(item, indent=indent) for item in prefix_items) + "]"
        items = schema.get("items")
        item_type = _typescript_type(items, indent=indent) if isinstance(items, Mapping) else "unknown"
        return f"ReadonlyArray<{item_type}>"
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}
        required = set(schema.get("required") or [])
        child_indent = indent + "  "
        rows = ["{"]
        for name, property_schema in properties.items():
            optional = "" if name in required else "?"
            rows.append(
                f"{child_indent}readonly {_typescript_property_name(str(name))}{optional}: "
                f"{_typescript_type(property_schema, indent=child_indent)};"
            )
        additional = schema.get("additionalProperties")
        if not properties and additional is not False:
            value_type = _typescript_type(additional, indent=child_indent) if isinstance(additional, Mapping) else "unknown"
            rows.append(f"{child_indent}readonly [key: string]: {value_type};")
        rows.append(f"{indent}}}")
        return "\n".join(rows)
    return "unknown"


def _contract_typescript() -> str:
    definitions: dict[str, dict[str, Any]] = {}
    for _key, model in sorted(CONTRACT_MODELS.items()):
        schema = model.model_json_schema(mode="serialization")
        for name, definition in sorted((schema.get("$defs") or {}).items()):
            normalized = copy.deepcopy(definition)
            previous = definitions.get(name)
            if previous is not None and previous != normalized:
                raise ValueError(f"incompatible duplicate JSON Schema definition: {name}")
            definitions[name] = normalized
        root = copy.deepcopy(schema)
        root.pop("$defs", None)
        name = str(root.get("title") or model.__name__)
        previous = definitions.get(name)
        if previous is not None and previous != root:
            raise ValueError(f"incompatible root/definition JSON Schema: {name}")
        definitions[name] = root

    rows = [
        "// Generated by scripts/export_noesis_core_schemas.py. Do not edit by hand.",
        "// Product contracts are immutable snapshots; unsupported major versions must fail closed.",
        "",
    ]
    for name, schema in sorted(definitions.items()):
        rendered = _typescript_type(schema)
        if rendered.startswith("{"):
            rows.append(f"export type {name} = {rendered};")
        else:
            rows.append(f"export type {name} = {rendered};")
        rows.append("")
    return "\n".join(rows)


def export_typescript(output_path: Path, *, check: bool) -> list[str]:
    output_path = output_path.resolve()
    expected = _contract_typescript()
    if check:
        if not output_path.exists() or output_path.read_text(encoding="utf-8") != expected:
            return [str(output_path)]
        return []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(expected, encoding="utf-8")
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Export deterministic Noesis product JSON schemas")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--typescript-output", type=Path, default=DEFAULT_TYPESCRIPT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mismatches = export_schemas(args.output, check=bool(args.check))
    mismatches.extend(export_typescript(args.typescript_output, check=bool(args.check)))
    if mismatches:
        for path in mismatches:
            print(f"schema out of date: {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
