#!/usr/bin/env python3
"""Verify MapAnything weights checksum against docs/ma-integration/weights.sha."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from services.mapanything_svc.server import verify_weights_checksum

if __name__ == '__main__':
    try:
        verify_weights_checksum()
    except Exception as exc:
        print(f"Verification failed: {exc}")
        sys.exit(1)
    print('MapAnything weights checksum verified.')
