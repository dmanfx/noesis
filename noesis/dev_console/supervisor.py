from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from noesis.ds8_preflight import REPO_ROOT
from noesis.dev_console.launch_spec import LaunchSpec
from noesis.dev_console.materialize import materialize_launch_pipeline
from noesis.dev_console.validator import validate_launch


class RuntimeSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.process: Optional[subprocess.Popen[str]] = None
        self.spec: Optional[LaunchSpec] = None
        self.started_at: Optional[float] = None
        self.materialized_pipeline: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self._log_handle: Optional[Any] = None
        self.last_validation: Optional[Dict[str, Any]] = None

    def is_running(self) -> bool:
        proc = self.process
        return proc is not None and proc.poll() is None

    def start(self, spec: LaunchSpec, *, allow_warnings: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self.is_running():
                raise RuntimeError("DS8 runtime is already running under the console supervisor")
            validation = validate_launch(spec)
            self.last_validation = validation
            if validation.get("blocking"):
                raise RuntimeError("Launch blocked by preflight validation")

            materialized = materialize_launch_pipeline(spec, dry_run=False)
            rel_pipeline = str(materialized.relative_to(REPO_ROOT)) if materialized.is_relative_to(REPO_ROOT) else str(materialized)
            argv = spec.to_argv(pipeline_config=rel_pipeline)
            spec.materialized_pipeline = rel_pipeline

            spec.launch_dir.mkdir(parents=True, exist_ok=True)
            log_path = spec.launch_dir / "runtime.log"
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write(f"\n[dev-console] launching at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write("[dev-console] argv: " + " ".join(argv) + "\n")
            proc = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                env=spec.to_env(os.environ),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid,
            )
            self.process = proc
            self.spec = spec
            self.started_at = time.time()
            self.materialized_pipeline = materialized
            self.log_path = log_path
            self._log_handle = log_handle
            return self.status()

    def stop(self, *, timeout: float = 12.0) -> Dict[str, Any]:
        with self._lock:
            proc = self.process
            if proc is None:
                return self.status()
            if proc.poll() is not None:
                self._close_log_handle()
                return self.status()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            except Exception:
                proc.send_signal(signal.SIGINT)

        deadline = time.time() + max(0.5, float(timeout))
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            time.sleep(0.2)

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.kill()
        self._close_log_handle()
        return self.status()

    def restart(self, spec: LaunchSpec) -> Dict[str, Any]:
        self.stop()
        return self.start(spec)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            proc = self.process
            running = proc is not None and proc.poll() is None
            returncode = None if proc is None else proc.poll()
            uptime = None
            if running and self.started_at is not None:
                uptime = max(0.0, time.time() - self.started_at)
            return {
                "running": running,
                "pid": None if proc is None else proc.pid,
                "returncode": returncode,
                "uptime_s": uptime,
                "started_at": self.started_at,
                "launch_id": None if self.spec is None else self.spec.launch_id,
                "materialized_pipeline": None if self.materialized_pipeline is None else str(self.materialized_pipeline),
                "log_path": None if self.log_path is None else str(self.log_path),
                "spec": None if self.spec is None else self.spec.to_dict(),
                "last_validation": self.last_validation,
            }

    def tail_log(self, *, lines: int = 200) -> Dict[str, Any]:
        path = self.log_path
        if path is None or not path.exists():
            return {"path": None if path is None else str(path), "lines": []}
        max_lines = max(1, min(2000, int(lines)))
        text_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"path": str(path), "lines": text_lines[-max_lines:]}

    def _close_log_handle(self) -> None:
        handle = self._log_handle
        self._log_handle = None
        if handle is not None:
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
