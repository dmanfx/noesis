#!/usr/bin/env python3
"""Evaluate a canonical snapshot against a reviewed map at its explicit recorded time."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noesis_core.contracts.world import WorldSnapshot
from noesis_core.scene_semantics import advisory_context, validate_map
from noesis_core.strict_json import strict_json_loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--snapshot-record", type=Path, required=True)
    parser.add_argument("--at-us", type=int, help="Default: snapshot's recorded publication time")
    args = parser.parse_args()
    for path in (args.map, args.snapshot_record):
        if path.stat().st_size > 2_000_000:
            raise ValueError("bounded JSON input exceeds 2 MB")
    scene = validate_map(strict_json_loads(args.map.read_bytes(), label="semantic map"))
    record = strict_json_loads(args.snapshot_record.read_bytes(), label="world snapshot record")
    snapshot = WorldSnapshot.model_validate(record.get("payload", record))
    if len(snapshot.entities) > 256:
        raise ValueError("bounded context demonstration admits at most 256 entities")
    at = args.at_us if args.at_us is not None else snapshot.published_at_us
    results = [advisory_context(scene, e.model_dump(mode="json"), at_us=at) for e in snapshot.entities]
    print(json.dumps({"snapshot_id": snapshot.snapshot_id, "evaluation_time_us": at,
                      "semantic_revision": scene.revision_sha256, "results": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
