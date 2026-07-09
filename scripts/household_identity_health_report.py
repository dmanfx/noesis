#!/usr/bin/env python3
"""Print household identity health from a live Noesis ReID REST endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Noesis REST base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()
    url = args.base_url.rstrip("/") + "/api/v1/reid/identity_health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"failed to fetch {url}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("Household identity health")
    print(f"  household_mode: {payload.get('household_mode')}")
    print(f"  residents: {payload.get('resident_count')}  visitors: {payload.get('visitor_count')}  provisional: {payload.get('provisional_count')}")
    print(f"  active_unique: {payload.get('active_unique')}  gallery_ids: {payload.get('gallery_ids')}")
    print(f"  mint_visitor: {payload.get('mint_visitor_count')}  promote_resident: {payload.get('promote_resident_count')}")
    print(f"  false_share_blocked: {payload.get('false_share_blocked_count')}")
    print(f"  overlap_grant/deny: {payload.get('overlap_permit_grant_count')}/{payload.get('overlap_permit_deny_count')}")
    print(f"  gallery_quality_reject: {payload.get('gallery_quality_reject_count')}  mnn_reject: {payload.get('mnn_reject_count')}")
    residents = payload.get("residents") or []
    if residents:
        print("  enrolled:")
        for row in residents:
            print(f"    - {row.get('stable_id')}: {row.get('display_name')} ({row.get('uuid')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
