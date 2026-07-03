from __future__ import annotations

import argparse
import json

import uvicorn

from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import materialize_launch_pipeline
from noesis.dev_console.presets import list_presets
from noesis.dev_console.server import create_app
from noesis.dev_console.validator import validate_launch


def main() -> int:
    parser = argparse.ArgumentParser(description="Noesis DS8 Dev Console")
    parser.add_argument("--host", default="127.0.0.1", help="Console bind host")
    parser.add_argument("--port", type=int, default=9090, help="Console bind port")
    parser.add_argument("--list-presets", action="store_true", help="List launch presets and exit")
    parser.add_argument("--validate", action="store_true", help="Validate the default launch spec and exit")
    parser.add_argument("--materialize", action="store_true", help="Materialize the default launch spec and exit")
    args = parser.parse_args()

    if args.list_presets:
        print(json.dumps([preset.to_dict() for preset in list_presets()], indent=2, sort_keys=True))  # type: ignore[union-attr]
        return 0
    if args.validate:
        print(json.dumps(validate_launch(LaunchSpec()), indent=2, sort_keys=True))
        return 0
    if args.materialize:
        path = materialize_launch_pipeline(LaunchSpec(), dry_run=True)
        print(str(path))
        return 0

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
