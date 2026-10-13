#!/usr/bin/env python3
"""Loopback-only review server. Writes only validated, revision-bound draft/export files."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from noesis_core.scene_semantics import advisory_context, diagnostics, export_accepted, seal_map, validate_map
from noesis_core.strict_json import strict_json_loads


class ReviewStore:
    def __init__(self, root):
        self.root = root.resolve()
        self.proposals = validate_map(json.loads((root / "proposals.json").read_text()))
        self.lock = threading.Lock()

    def current(self):
        return validate_map(json.loads((self.root / "reviewed.json").read_text()))

    def state(self):
        with self.lock:
            scene = self.current()
            fixtures = json.loads((self.root / "recorded-context.json").read_text())
            for f in fixtures:
                f["context"] = advisory_context(scene, f["entity"], at_us=f["evaluation_time_us"])
            return {"map": scene.model_dump(mode="json"), "diagnostics": diagnostics(scene), "fixtures": fixtures}

    def save(self, request):
        if set(request) != {"base_revision", "elements", "actor", "reason"}:
            raise ValueError("unexpected/missing edit fields")
        actor, reason = request["actor"], request["reason"]
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 96:
            raise ValueError("review actor required (max 96 chars)")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("review reason required (max 2000 chars)")
        with self.lock:
            old = self.current()
            if request["base_revision"] != old.revision_sha256:
                raise RuntimeError("Draft changed since load; reload before saving")
            payload = old.model_dump(mode="json")
            incoming = copy.deepcopy(request["elements"])
            if not isinstance(incoming, list) or len(incoming) > 256:
                raise ValueError("bounded elements list required")
            old_by = {e.element_id: e.model_dump(mode="json") for e in old.elements}
            original_by = {e.element_id: e for e in self.proposals.elements}
            incoming_ids = {e.get("element_id") for e in incoming if isinstance(e, dict)}
            if not set(old_by).issubset(incoming_ids):
                raise ValueError("Keep original history; reject elements instead of deleting")
            changes = []
            for e in incoming:
                if not isinstance(e, dict):
                    raise ValueError("invalid element")
                eid = e.get("element_id")
                previous = old_by.get(eid)
                if previous != e:
                    e["review_actor"] = actor.strip()
                    e["review_reason"] = reason.strip()
                    if eid in original_by:
                        if e.get("kind") != original_by[eid].kind or e.get("source_kind") != "model_proposal":
                            raise ValueError("Original element kind/provenance is immutable")
                    elif e.get("source_kind") != "manual_annotation":
                        raise ValueError("New geometry must declare manual annotation")
                    changes.append({"element_id": eid, "before": previous, "after": e})
            payload["elements"] = incoming
            scene = seal_map(payload)
            if not changes:
                return {"map": old.model_dump(mode="json"), "saved": False, "message": "No changes"}
            stamp = datetime.now(timezone.utc).isoformat()
            # One immutable full draft snapshot includes its audit; latest pointer is atomic.
            event = {"saved_at_utc": stamp, "actor": actor.strip(), "reason": reason.strip(),
                     "base_revision": old.revision_sha256, "result_revision": scene.revision_sha256, "changes": changes}
            history = self.root / "drafts" / (stamp.replace(":", "-") + "_" + scene.revision_sha256 + ".json")
            with history.open("x") as f:
                json.dump({"map": scene.model_dump(mode="json"), "audit": event}, f, indent=2, allow_nan=False)
                f.write("\n")
            temporary = self.root / "reviewed.json.tmp"
            temporary.write_text(json.dumps(scene.model_dump(mode="json"), indent=2, allow_nan=False) + "\n")
            temporary.replace(self.root / "reviewed.json")
            return {"saved": True, "map": scene.model_dump(mode="json"), "diagnostics": diagnostics(scene), "audit": event}

    def export(self):
        with self.lock:
            scene = export_accepted(self.current())
            path = self.root / "exports" / (scene.revision_sha256 + ".json")
            text = json.dumps(scene.model_dump(mode="json"), indent=2, allow_nan=False) + "\n"
            if path.exists() and path.read_text() != text:
                raise ValueError("Export revision collision")
            if not path.exists():
                with path.open("x") as f:
                    f.write(text)
            (self.root / "exports/accepted-map.json").write_text(text)
            return {"map": scene.model_dump(mode="json"), "download_name": "living-room-accepted-semantics.json"}


class Handler(SimpleHTTPRequestHandler):
    ASSETS = {"package.json", "dimensions.json", "floor-rise.json", "validation.json", "photo.png",
              "cloud.xyz.f32", "cloud.rgb.u8", "cloud-full.xyz.f32", "cloud-full.rgb.u8", "proposals.json",
              "stability/seed-comparison.json", "evidence/seed-stability.png"}

    def __init__(self, *args, store, three_root, **kwargs):
        self.store, self.three_root = store, three_root.resolve()
        super().__init__(*args, directory=str(store.root), **kwargs)

    def log_message(self, fmt, *args):
        # No source identities, private credentials or payloads in access logs.
        pass

    def allowed_host(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def translate_path(self, path):
        route = unquote(urlsplit(path).path)
        if route in ("/", "/index.html", "/viewer.js", "/viewer.css"):
            return str(Path(__file__).parent / "viewer" / (route.lstrip("/") or "index.html"))
        if route.startswith("/vendor/"):
            target = (self.three_root / route.removeprefix("/vendor/")).resolve()
            if target.is_relative_to(self.three_root) and target.is_file() and target.suffix == ".js":
                return str(target)
        if route.lstrip("/") in self.ASSETS:
            return str(self.store.root / route.lstrip("/"))
        return str(self.store.root / "__unavailable__")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        super().end_headers()

    def json_response(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.allowed_host():
            self.send_error(403, "Loopback Host required")
            return
        if self.path == "/api/state":
            self.json_response(self.store.state())
        else:
            super().do_GET()

    def do_POST(self):
        origin = self.headers.get("Origin")
        expected_origin = f"http://{self.headers.get('Host')}"
        if not self.allowed_host() or origin != expected_origin or self.headers.get("Content-Type") != "application/json":
            self.json_response({"error": "Same-origin JSON edit required"}, 403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2_000_000:
                raise ValueError("Request size must be between 1 byte and 2 MB")
            payload = strict_json_loads(self.rfile.read(size), label="semantic review edit")
            if self.path == "/api/save":
                self.json_response(self.store.save(payload))
            elif self.path == "/api/export" and payload == {}:
                self.json_response(self.store.export())
            else:
                self.json_response({"error": "Unknown request"}, 404)
        except RuntimeError as exc:
            self.json_response({"error": str(exc)}, 409)
        except (ValueError, TypeError, KeyError) as exc:
            self.json_response({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--three-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8873)
    args = parser.parse_args()
    store = ReviewStore(args.package_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, store=store, three_root=args.three_root))
    print(f"Semantic review: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
