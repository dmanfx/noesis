"""Serialize semantic captures and publish only complete, validated results."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .assets import REPO_ROOT, SUPPORTED_SIZES


SAFE_CAPTURE_ID = re.compile(r"^[A-Za-z0-9._~-]{1,160}$")
CAMERAS = {
    0: ("living-room", "Living Room"),
    1: ("kitchen", "Kitchen"),
    2: ("family-room", "Family Room"),
}
ARTIFACT_KEYS = {
    "raw": "raw_path",
    "class-map": "class_map_path",
    "masked": "masked_path",
}


class SemanticCaptureError(RuntimeError):
    pass


class SemanticCaptureBusy(SemanticCaptureError):
    pass


class SemanticCaptureTimeout(SemanticCaptureError):
    pass


def _normalize_model(value: str) -> str:
    model = str(value or "").strip().lower()
    if model not in SUPPORTED_SIZES:
        raise ValueError(f"Semantic model must be one of {SUPPORTED_SIZES}")
    return model


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class SemanticCaptureManager:
    def __init__(
        self,
        *,
        capture_root: Path | None = None,
        runner_timeout_s: float = 45.0,
        command_runner: Any = subprocess.run,
    ) -> None:
        configured = str(os.environ.get("NOESIS_SEMANTIC_CAPTURE_ROOT", "") or "").strip()
        default_root = Path(os.environ.get("NOESIS_BUILD_DIR", REPO_ROOT / "build")) / "semantic-seg" / "captures"
        self.capture_root = Path(capture_root or (Path(configured) if configured else default_root)).expanduser().resolve()
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.runner_timeout_s = max(5.0, float(runner_timeout_s))
        self.command_runner = command_runner
        self._capture_lock = threading.Lock()

    def capture(self, model: str) -> dict[str, Any]:
        normalized = _normalize_model(model)
        if not self._capture_lock.acquire(blocking=False):
            raise SemanticCaptureBusy("Another semantic capture is already running")
        capture_id = f"capture-{time.time_ns()}-{uuid.uuid4().hex[:10]}"
        output_dir = self.capture_root / capture_id / f"yolo26{normalized}"
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable,
                "-m",
                "noesis.semantic_capture.main",
                "--size",
                normalized,
                "--duration",
                "30",
                "--warmup-frames",
                "8",
                "--output-dir",
                str(output_dir),
            ]
            try:
                result = self.command_runner(
                    command,
                    cwd=str(REPO_ROOT),
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    timeout=self.runner_timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise SemanticCaptureTimeout(
                    f"Semantic capture exceeded {self.runner_timeout_s:.0f} seconds"
                ) from exc
            if int(result.returncode) != 0:
                detail = str(result.stderr or result.stdout or "semantic runner failed").strip()
                raise SemanticCaptureError(detail[-2000:])
            record, public = self._validate_and_publish(capture_id, normalized, output_dir)
            _atomic_json(output_dir / "capture_record.json", record)
            _atomic_json(self.capture_root / f"latest-{normalized}.json", record)
            return public
        except Exception:
            shutil.rmtree(self.capture_root / capture_id, ignore_errors=True)
            raise
        finally:
            self._capture_lock.release()

    def _validate_and_publish(
        self,
        capture_id: str,
        model: str,
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        summary_path = output_dir / "capture_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SemanticCaptureError("Semantic runner did not publish a valid summary") from exc
        rows = summary.get("captures") if isinstance(summary, dict) else None
        if not isinstance(rows, list) or len(rows) != 3:
            raise SemanticCaptureError("Semantic runner did not publish exactly three camera results")
        source_ids = [row.get("source_id") for row in rows if isinstance(row, dict)]
        if sorted(source_ids) != sorted(CAMERAS):
            raise SemanticCaptureError("Semantic runner did not publish one result for each canonical camera")
        artifacts: list[dict[str, Any]] = []
        cameras: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("source_id"), int):
                raise SemanticCaptureError("Semantic runner camera result is malformed")
            source_id = int(row["source_id"])
            if source_id not in CAMERAS:
                raise SemanticCaptureError(f"Unexpected semantic source ID {source_id}")
            camera_id, label = CAMERAS[source_id]
            relative: dict[str, str] = {}
            for artifact, key in ARTIFACT_KEYS.items():
                path = Path(str(row.get(key) or "")).expanduser().resolve()
                if output_dir not in path.parents or not path.is_file() or path.stat().st_size <= 0:
                    raise SemanticCaptureError(f"Semantic {artifact} artifact is missing or outside its capture")
                relative[artifact] = path.relative_to(output_dir).as_posix()
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
            if width <= 0 or height <= 0:
                raise SemanticCaptureError("Semantic class-map dimensions are invalid")
            top_classes = row.get("top_classes")
            if not isinstance(top_classes, list):
                raise SemanticCaptureError("Semantic class summary is invalid")
            base_url = f"/api/v1/semantic-seg/captures/{capture_id}/{camera_id}"
            cameras.append({
                "id": camera_id,
                "label": label,
                "source_id": source_id,
                "width": width,
                "height": height,
                "top_classes": top_classes,
                "raw_url": f"{base_url}/raw",
                "class_map_url": f"{base_url}/class-map",
                "masked_url": f"{base_url}/masked",
            })
            artifacts.append({"id": camera_id, "files": relative})
        cameras.sort(key=lambda item: int(item["source_id"]))
        source_order = {camera_id: source_id for source_id, (camera_id, _label) in CAMERAS.items()}
        artifacts.sort(key=lambda item: source_order[str(item["id"])])
        captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        public = {
            "contract": "noesis.semantic_seg.capture",
            "contract_version": 1,
            "capture_id": capture_id,
            "model": model,
            "model_name": f"yolo26{model}-sem-ade20k",
            "captured_at": captured_at,
            "cameras": cameras,
        }
        return {
            **public,
            "output_dir": str(output_dir),
            "artifacts": artifacts,
        }, public

    def latest(self, model: str) -> dict[str, Any]:
        normalized = _normalize_model(model)
        path = self.capture_root / f"latest-{normalized}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No completed semantic capture exists for model {normalized}")
        record = json.loads(path.read_text(encoding="utf-8"))
        return self._public_record(record, expected_model=normalized)

    def artifact_path(self, capture_id: str, camera_id: str, artifact: str) -> Path:
        if not SAFE_CAPTURE_ID.fullmatch(str(capture_id or "")):
            raise ValueError("Semantic capture ID is invalid")
        if camera_id not in {pair[0] for pair in CAMERAS.values()} or artifact not in ARTIFACT_KEYS:
            raise ValueError("Semantic artifact selector is invalid")
        capture_dir = self.capture_root / capture_id
        model_dirs = [path for path in capture_dir.iterdir() if path.is_dir()] if capture_dir.is_dir() else []
        if len(model_dirs) != 1:
            raise FileNotFoundError("Semantic capture does not exist")
        record_path = model_dirs[0] / "capture_record.json"
        if not record_path.is_file():
            raise FileNotFoundError("Semantic capture does not exist")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        output_dir = Path(str(record.get("output_dir") or "")).resolve()
        if output_dir != record_path.parent.resolve():
            raise SemanticCaptureError("Semantic capture record output path is invalid")
        row = next((item for item in record.get("artifacts", []) if item.get("id") == camera_id), None)
        relative = row.get("files", {}).get(artifact) if isinstance(row, dict) else None
        path = (output_dir / str(relative or "")).resolve()
        if output_dir not in path.parents or not path.is_file():
            raise FileNotFoundError("Semantic artifact does not exist")
        return path

    @staticmethod
    def _public_record(record: Any, *, expected_model: str) -> dict[str, Any]:
        if not isinstance(record, dict) or record.get("model") != expected_model:
            raise SemanticCaptureError("Semantic latest record is invalid")
        return {key: record[key] for key in (
            "contract", "contract_version", "capture_id", "model", "model_name", "captured_at", "cameras"
        )}
