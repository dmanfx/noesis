from __future__ import annotations

import os
import socket

import uvicorn


def _lan_ip() -> str | None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))
        address = str(probe.getsockname()[0])
        return address if address and not address.startswith("127.") else None
    except OSError:
        return None
    finally:
        probe.close()


def main() -> None:
    host = os.environ.get("NOESIS_PHONE_SCAN_HOST", "0.0.0.0")
    port = int(os.environ.get("NOESIS_PHONE_SCAN_PORT", "8788"))
    address = _lan_ip()
    print("Noesis multi-view phone scan is starting.", flush=True)
    if address:
        print(f"Phone URL: http://{address}:{port}", flush=True)
    else:
        print(f"Open this machine's LAN address on port {port}.", flush=True)
    print("Keep this terminal running while recording, uploading, and reviewing.", flush=True)
    uvicorn.run(
        "tools.mapanything_phone_scan.app:app",
        host=host,
        port=port,
        log_level=os.environ.get("NOESIS_PHONE_SCAN_LOG_LEVEL", "info"),
        access_log=False,
    )


if __name__ == "__main__":
    main()
