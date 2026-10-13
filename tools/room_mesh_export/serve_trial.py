#!/usr/bin/env python3
"""Serve an isolated room-mesh trial and the explicitly selected Three.js build."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, trial_root, three_root, source_glb, menon_loader, **kwargs):
        self.trial_root = trial_root
        self.three_root = three_root
        self.source_glb = source_glb
        self.menon_loader = menon_loader
        super().__init__(*args, directory=trial_root, **kwargs)

    def translate_path(self, path):
        route = unquote(urlsplit(path).path)
        if route == "/source.glb":
            return str(self.source_glb)
        if route == "/menon-loader.js":
            return str(self.menon_loader)
        if route.startswith("/vendor/"):
            root, relative = self.three_root, route.removeprefix("/vendor/")
        elif route in ("/", "/index.html", "/viewer.js", "/viewer.css"):
            root = Path(__file__).parent / "viewer"
            relative = route.removeprefix("/") or "index.html"
        else:
            root, relative = self.trial_root, route.removeprefix("/")
            # Only trial assets/configuration are exposed, not build dependencies.
            if relative.split("/", 1)[0] in ("js-tools", "python-deps"):
                return str(self.trial_root / "__unavailable__")
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            return str(self.trial_root / "__unavailable__")
        return str(target)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial-root", "three-root", "source-glb", "menon-loader"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=8842)
    args = parser.parse_args()
    handler = partial(Handler, trial_root=args.trial_root.resolve(), three_root=args.three_root.resolve(),
                      source_glb=args.source_glb.resolve(), menon_loader=args.menon_loader.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Room mesh trial: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
