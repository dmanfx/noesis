from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict


def _request_json(url: str, *, method: str = "GET", timeout: float = 4.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310 - local operator URL only
            body = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                payload = {"raw": body}
            return {"ok": True, "status": response.status, "url": url, "payload": payload}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "url": url, "error": body}
    except Exception as exc:
        return {"ok": False, "status": None, "url": url, "error": str(exc)}


def depth_refresh(*, rest_host: str, rest_port: int, seconds: int) -> Dict[str, Any]:
    query = urllib.parse.urlencode({"seconds": int(seconds)})
    url = f"http://{rest_host}:{int(rest_port)}/api/v1/depth/refresh?{query}"
    return _request_json(url, method="POST")


def runtime_health(*, rest_host: str, rest_port: int) -> Dict[str, Any]:
    return _request_json(f"http://{rest_host}:{int(rest_port)}/api/v1/health")
