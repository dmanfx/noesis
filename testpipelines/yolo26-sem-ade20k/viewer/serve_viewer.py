"""Serve the no-build semantic pixel inspector from the repository root."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWER_PATH = "/testpipelines/yolo26-sem-ade20k/viewer/index.html"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the YOLO26 semantic pixel inspector")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    handler = partial(NoCacheHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Semantic viewer: http://{args.bind}:{args.port}{VIEWER_PATH}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
