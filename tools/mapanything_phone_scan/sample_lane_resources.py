#!/usr/bin/env python3
"""Bounded, read-only GPU/RAM sampling for one explicit offline lane."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--period-seconds", type=float, default=5)
    parser.add_argument("--max-samples", type=int, default=720)
    args = parser.parse_args()
    if not 1 <= args.period_seconds <= 30 or not 1 <= args.max_samples <= 1440:
        raise ValueError("sampling interval/count exceed bounds")
    fields = ["utc", "stage", "gpu_used_mib", "gpu_free_mib", "gpu_util_percent", "ram_available_kib"]
    with args.output_file.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for _ in range(args.max_samples):
            manifest = json.loads((args.lane_root / "lane_manifest.json").read_text())
            stages = manifest.get("stages", [])
            stage = stages[-1]["name"] if stages else "handoff"
            gpu = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.free,utilization.gpu",
                                  "--format=csv,noheader,nounits"], check=True,
                                 capture_output=True, text=True, timeout=5).stdout.strip().split(",")
            mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            writer.writerow(dict(zip(fields, [datetime.now(timezone.utc).isoformat(), stage,
                *[int(value) for value in gpu], int(mem["MemAvailable"].split()[0])], strict=True)))
            output.flush()
            lease_path = args.lane_root / "runtime_lease.json"
            lease = json.loads(lease_path.read_text()) if lease_path.exists() else {}
            if manifest["status"] != "running" or lease.get("restore_passed") is True:
                break
            time.sleep(args.period_seconds)
    print(f"Saved sampled resource evidence to {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
