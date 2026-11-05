"""Small, recorded adapters around the installed World Labs CLI.

Keys stay with the CLI. Requests, signed URLs, responses and media belong only
in the private external output root, never in the repository reports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


class BenchError(RuntimeError):
    pass


def private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def redact(value: str) -> str:
    value = re.sub(r"wl_(?:live|test)_[A-Za-z0-9_-]+", "[REDACTED]", value)
    key = os.environ.get("WLT_API_KEY", "").strip()
    return value.replace(key, "[REDACTED]") if key else value


def resolve_command(local: dict) -> list[str]:
    if os.environ.get("NOESIS_WORLDLABS_COMMAND"):
        return shlex.split(os.environ["NOESIS_WORLDLABS_COMMAND"])
    if local.get("worldlabs_command"):
        return local["worldlabs_command"]
    found = shutil.which("worldlabs")
    if found:
        executable = Path(found)
        node = executable.parent / "node"
        return [str(node), str(executable)] if node.exists() else [found]
    # nvm installs are deliberately isolated from the application's Node choice.
    versions = Path.home() / ".nvm/versions/node"
    candidates = []
    for executable in versions.glob("v*/bin/worldlabs"):
        version = tuple(int(x) for x in executable.parents[1].name[1:].split("."))
        if version[0] >= 24:
            candidates.append((version, executable))
    if candidates:
        executable = max(candidates)[1]
        return [str(executable.parent / "node"), str(executable)]
    raise BenchError("World Labs CLI missing. Install under Node 24+: npm install -g @worldlabsai/cli")


class WorldLabs:
    def __init__(self, local: dict):
        self.command = resolve_command(local)

    def runtime(self) -> dict:
        node = self.command[0] if Path(self.command[0]).name == "node" else shutil.which("node")
        if not node:
            raise BenchError("Node.js is missing; World Labs requires Node 24+")
        version = subprocess.check_output([node, "--version"], text=True, timeout=10).strip()
        if int(version.lstrip("v").split(".")[0]) < 24:
            raise BenchError("Selected World Labs CLI needs Node 24+; configure NOESIS_WORLDLABS_COMMAND")
        return {"version": version, "executable": node}

    def call(self, args: list[str], record: Path | None = None,
             timeout: int = 90, json_output: bool = True) -> object:
        started = time.monotonic()
        command = self.command + args + (["--json"] if json_output else [])
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if record:
                private_json(record.with_suffix(".error.json"), {
                    "kind": "local_timeout", "seconds": timeout,
                    "message": "Server work may continue. Resume this run; do not resubmit with a new key."})
            raise BenchError("CLI timeout; server work may continue. Use resume with the same run ID.") from exc
        stdout, stderr = redact(completed.stdout), redact(completed.stderr)
        try:
            value = json.loads(stdout) if json_output and stdout.strip() else stdout.strip()
        except json.JSONDecodeError as exc:
            if record:
                private_json(record.with_suffix(".error.json"), {
                    "kind": "invalid_json", "stdout": stdout, "stderr": stderr,
                    "returncode": completed.returncode, "arguments": args})
            raise BenchError("World Labs CLI returned invalid JSON") from exc
        if record:
            private_json(record, value)
            private_json(record.with_suffix(".cli.json"), {
                "arguments": args, "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started, "stderr": stderr})
        if completed.returncode:
            if isinstance(value, dict) and value.get("done") and value.get("error"):
                raise BenchError("World Labs operation failed; inspect the private operation JSON.")
            raise BenchError(stderr.strip() or "World Labs CLI failed; inspect private command records.")
        return value

    def auth(self) -> dict:
        result = self.call(["auth", "status"], timeout=40)
        if not isinstance(result, dict) or result.get("authenticated") is not True:
            raise BenchError("Atlas authentication missing. Run worldlabs auth login privately.")
        return result

    def upload(self, source: Path, record: Path, mime: str | None = None) -> dict:
        args = ["assets", "upload", str(source)]
        if mime:
            args += ["--mime", mime]
        result = self.call(args, record, timeout=650)
        if not isinstance(result, dict) or not result.get("id"):
            raise BenchError("Upload did not return an asset ID")
        return result

    def submit(self, task: str, request: Path, key: str, record: Path) -> dict:
        # No --wait: persist the accepted operation before any long wait.
        return self.call(["tasks", "run", task, "--request", str(request),
                          "--idempotency-key", key, "--timeout", "55"], record, timeout=65)

    def poll(self, operation_id: str, record: Path) -> dict:
        return self.call(["operations", "get", operation_id, "--timeout", "40"], record, timeout=50)

    def download(self, asset_id: str, destination: Path, record: Path) -> None:
        self.call(["assets", "download", asset_id, "--out", str(destination)], record, timeout=650)
