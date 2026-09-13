"""Bounded background work and durable native-capture upload receipts."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable


ASYNC_VIDEO_PROBE_TIMEOUT_S = 1800
UPLOAD_RECEIPT_SCHEMA = "noesis.phone_capture.upload_receipt.v1"
LOG = logging.getLogger(__name__)


class CaptureUploadBusy(ValueError):
    pass


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


class CaptureUploadQueue:
    """One importer and at most two waiting jobs; requests do not own workers.

    The identity index refers only to scan state. The original archive, receipt,
    and import status are retained together in that scan's directory. Recovery
    marks interrupted scans failed; resending the identical archive explicitly
    retries validation without inventing an association or duplicating a job.
    """

    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.index_root = storage_root / ".capture-upload-index"
        self.index_root.mkdir(exist_ok=True)
        self._guard = threading.Lock()
        self._admitted: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CaptureImport")

    def _index_path(self, capture_id: str, session_id: str | None) -> Path:
        identity = json.dumps([session_id, capture_id], separators=(",", ":")).encode()
        return self.index_root / (hashlib.sha256(identity).hexdigest() + ".json")

    def lookup(self, capture_id: str, session_id: str | None) -> str | None:
        path = self._index_path(capture_id, session_id)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value["scan_id"])

    def forget(self, capture_id: str, session_id: str | None, scan_id: str) -> None:
        if self.lookup(capture_id, session_id) == scan_id:
            self._index_path(capture_id, session_id).unlink()
            sync_directory(self.index_root)

    def reserve(self, scan_id: str) -> bool:
        with self._guard:
            if scan_id in self._admitted:
                return False
            if len(self._admitted) >= 3:
                raise CaptureUploadBusy("Capture import queue is full; retry after a pending import finishes")
            self._admitted.add(scan_id)
            return True

    def release(self, scan_id: str) -> None:
        with self._guard:
            self._admitted.discard(scan_id)

    def submit(self, scan_id: str, operation: Callable[[], Any]) -> None:
        def run() -> None:
            try:
                operation()
            except Exception:
                LOG.exception("Stored capture import failed for scan %s", scan_id)
            finally:
                self.release(scan_id)
        try:
            self._executor.submit(run)
        except BaseException:
            self.release(scan_id)
            raise

    def persist(
        self, temporary: Path, scan_dir: Path, state: dict[str, Any],
        write_state: Callable[[str, dict[str, Any]], None],
    ) -> Path:
        """Flush archive, receipt state, and identity index before acknowledging."""
        scan_dir.mkdir(exist_ok=True)
        archive = scan_dir / "capture_upload.archive"
        # A retry has already been compared with the durable receipt digest.
        # Keep the first retained archive rather than replace acknowledged data.
        if archive.exists():
            temporary.unlink(missing_ok=True)
        else:
            sync_file(temporary)
            os.replace(temporary, archive)
        sync_directory(scan_dir)
        write_state(state["id"], state)
        self.sync_state(scan_dir)
        receipt = state["upload_receipt"]
        index = self._index_path(receipt["capture_id"], receipt["companion_session_id"])
        staging = index.with_suffix(".json.tmp")
        with staging.open("w", encoding="utf-8") as handle:
            json.dump({"scan_id": state["id"]}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, index)
        sync_directory(self.index_root)
        sync_directory(self.storage_root)
        return archive

    @staticmethod
    def sync_state(scan_dir: Path) -> None:
        sync_file(scan_dir / "scan_state.json")
        sync_directory(scan_dir)

    def shutdown(self) -> None:
        # Admitted work outlives the originating HTTP request and a graceful
        # app shutdown. An actual process interruption leaves the archive intact.
        self._executor.shutdown(wait=False, cancel_futures=False)
