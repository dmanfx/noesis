"""Closed-world household identity helpers (Phase 1).

Residents: sticky IDs 1..N from enrollment table.
Visitors: recycled pool VISITOR_ID_MIN..VISITOR_ID_MAX.
Provisional: ephemeral display IDs, not gallery-persisted until confirmed.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VISITOR_ID_MIN = 1000
VISITOR_ID_MAX = 1031
PROVISIONAL_ID_MIN = 9000
PROVISIONAL_ID_MAX = 9099
RESIDENT_ID_MIN = 1


@dataclass
class ResidentRecord:
    uuid: str
    stable_id: int
    display_name: str
    created_ts: float
    embedding_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "stable_id": int(self.stable_id),
            "display_name": str(self.display_name),
            "created_ts": float(self.created_ts),
            "embedding_count": int(self.embedding_count),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Optional["ResidentRecord"]:
        try:
            return cls(
                uuid=str(raw["uuid"]),
                stable_id=int(raw["stable_id"]),
                display_name=str(raw.get("display_name", "")),
                created_ts=float(raw.get("created_ts", time.time())),
                embedding_count=int(raw.get("embedding_count", 0)),
            )
        except Exception:
            return None


def identity_kind_for_sid(
    sid: int,
    *,
    resident_ids: Optional[Set[int]] = None,
) -> str:
    sid_int = int(sid)
    if sid_int >= PROVISIONAL_ID_MIN:
        return "provisional"
    if resident_ids and sid_int in resident_ids:
        return "resident"
    if RESIDENT_ID_MIN <= sid_int < VISITOR_ID_MIN:
        return "resident"
    if VISITOR_ID_MIN <= sid_int <= VISITOR_ID_MAX:
        return "visitor"
    return "provisional"


def identity_state_for_kind(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k in ("resident", "visitor", "provisional", "handoff"):
        return k
    return "provisional"


class ResidentRegistry:
    """Enrollment table persisted under ~/.noesis/household/residents.json."""

    def __init__(self, path: str) -> None:
        self.path = os.path.expanduser(str(path))
        self.residents_by_sid: Dict[int, ResidentRecord] = {}
        self.uuid_by_sid: Dict[int, str] = {}
        self.sid_by_uuid: Dict[str, int] = {}
        self._next_resident_id = RESIDENT_ID_MIN
        self.load()

    def load(self) -> None:
        self.residents_by_sid.clear()
        self.uuid_by_sid.clear()
        self.sid_by_uuid.clear()
        self._next_resident_id = RESIDENT_ID_MIN
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.warning("failed to load residents from %s", self.path, exc_info=True)
            return
        if not isinstance(data, dict):
            return
        try:
            self._next_resident_id = max(RESIDENT_ID_MIN, int(data.get("next_resident_id", RESIDENT_ID_MIN)))
        except Exception:
            self._next_resident_id = RESIDENT_ID_MIN
        rows = data.get("residents", [])
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            rec = ResidentRecord.from_dict(row)
            if rec is None or rec.stable_id < RESIDENT_ID_MIN or rec.stable_id >= VISITOR_ID_MIN:
                continue
            self.residents_by_sid[int(rec.stable_id)] = rec
            self.uuid_by_sid[int(rec.stable_id)] = str(rec.uuid)
            self.sid_by_uuid[str(rec.uuid)] = int(rec.stable_id)
            if int(rec.stable_id) >= self._next_resident_id:
                self._next_resident_id = int(rec.stable_id) + 1

    def save(self) -> None:
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                "version": 1,
                "next_resident_id": int(self._next_resident_id),
                "residents": [rec.to_dict() for rec in sorted(self.residents_by_sid.values(), key=lambda r: r.stable_id)],
            }
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception:
            logger.warning("failed to save residents to %s", self.path, exc_info=True)

    def resident_ids(self) -> Set[int]:
        return set(self.residents_by_sid.keys())

    def list_residents(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in sorted(self.residents_by_sid.values(), key=lambda r: r.stable_id):
            out.append(dict(rec.to_dict()))
        return out

    def enroll(
        self,
        *,
        display_name: str,
        stable_id: Optional[int] = None,
        visitor_id: Optional[int] = None,
        embedding_count: int = 0,
        now_ts: Optional[float] = None,
    ) -> ResidentRecord:
        now = float(now_ts if now_ts is not None else time.time())
        name = str(display_name or "").strip()
        if not name:
            raise ValueError("display_name required")

        target_sid: Optional[int] = None
        if stable_id is not None:
            target_sid = int(stable_id)
        elif visitor_id is not None:
            target_sid = int(visitor_id)

        bind_from_sid: Optional[int] = None
        if target_sid is not None:
            if target_sid < RESIDENT_ID_MIN or target_sid > VISITOR_ID_MAX:
                raise ValueError("stable_id must be in resident or visitor range for enrollment bind")
            if VISITOR_ID_MIN <= int(target_sid) <= VISITOR_ID_MAX:
                bind_from_sid = int(target_sid)
            existing = self.residents_by_sid.get(int(target_sid))
            if existing is not None and RESIDENT_ID_MIN <= int(target_sid) < VISITOR_ID_MIN:
                existing.display_name = name
                if embedding_count > 0:
                    existing.embedding_count = int(embedding_count)
                self.save()
                return existing

        new_sid = int(self._next_resident_id)
        while new_sid in self.residents_by_sid:
            new_sid += 1
        if new_sid >= VISITOR_ID_MIN:
            raise RuntimeError("resident ID space exhausted")

        if target_sid is not None and RESIDENT_ID_MIN <= int(target_sid) < VISITOR_ID_MIN:
            new_sid = int(target_sid)

        rec = ResidentRecord(
            uuid=str(uuid.uuid4()),
            stable_id=int(new_sid),
            display_name=name,
            created_ts=now,
            embedding_count=int(max(0, embedding_count)),
        )
        self.residents_by_sid[int(new_sid)] = rec
        self.uuid_by_sid[int(new_sid)] = rec.uuid
        self.sid_by_uuid[rec.uuid] = int(new_sid)
        if int(new_sid) >= self._next_resident_id:
            self._next_resident_id = int(new_sid) + 1
        self.save()
        return rec

    def display_name_for(self, sid: int) -> Optional[str]:
        rec = self.residents_by_sid.get(int(sid))
        if rec is None:
            return None
        return str(rec.display_name)

    def uuid_for(self, sid: int) -> Optional[str]:
        return self.uuid_by_sid.get(int(sid))

    def get_by_uuid(self, resident_uuid: str) -> Optional[ResidentRecord]:
        sid = self.sid_by_uuid.get(str(resident_uuid))
        if sid is None:
            return None
        return self.residents_by_sid.get(int(sid))

    def patch(
        self,
        resident_uuid: str,
        *,
        display_name: Optional[str] = None,
        embedding_count: Optional[int] = None,
    ) -> ResidentRecord:
        rec = self.get_by_uuid(resident_uuid)
        if rec is None:
            raise ValueError(f"unknown resident uuid: {resident_uuid}")
        if display_name is not None:
            name = str(display_name).strip()
            if not name:
                raise ValueError("display_name required")
            rec.display_name = name
        if embedding_count is not None:
            rec.embedding_count = int(max(0, embedding_count))
        self.save()
        return rec

    def delete(self, resident_uuid: str) -> ResidentRecord:
        rec = self.get_by_uuid(resident_uuid)
        if rec is None:
            raise ValueError(f"unknown resident uuid: {resident_uuid}")
        sid = int(rec.stable_id)
        self.residents_by_sid.pop(sid, None)
        self.uuid_by_sid.pop(sid, None)
        self.sid_by_uuid.pop(str(resident_uuid), None)
        self.save()
        return rec


class _SidPool:
    """Small heap-backed SID pool for visitor or provisional ranges."""

    def __init__(self, id_min: int, id_max: int) -> None:
        self.id_min = int(id_min)
        self.id_max = int(id_max)
        self._free: List[int] = []
        self._free_set: Set[int] = set()
        self._used: Set[int] = set()

    def _in_range(self, sid: int) -> bool:
        return self.id_min <= int(sid) <= self.id_max

    def load_free(self, values: List[int]) -> None:
        for raw in values:
            try:
                sid = int(raw)
            except Exception:
                continue
            if not self._in_range(sid):
                continue
            if sid in self._free_set or sid in self._used:
                continue
            heapq.heappush(self._free, sid)
            self._free_set.add(sid)

    def free_list(self) -> List[int]:
        return sorted(int(s) for s in self._free_set)

    def alloc(self) -> int:
        while self._free:
            sid = int(heapq.heappop(self._free))
            self._free_set.discard(sid)
            if sid in self._used:
                continue
            self._used.add(sid)
            return sid
        for candidate in range(self.id_min, self.id_max + 1):
            if candidate not in self._used:
                self._used.add(int(candidate))
                return int(candidate)
        raise RuntimeError(f"SID pool exhausted ({self.id_min}..{self.id_max})")

    def release(self, sid: int) -> None:
        sid_int = int(sid)
        if not self._in_range(sid_int):
            return
        self._used.discard(sid_int)
        if sid_int in self._free_set:
            return
        heapq.heappush(self._free, sid_int)
        self._free_set.add(sid_int)

    def mark_used(self, sid: int) -> None:
        sid_int = int(sid)
        if not self._in_range(sid_int):
            return
        self._used.add(sid_int)
        if sid_int in self._free_set:
            self._free_set.discard(sid_int)
            self._free = [s for s in self._free if int(s) != sid_int]
            heapq.heapify(self._free)


class VisitorPool:
    def __init__(
        self,
        *,
        pool_file: str,
        id_min: int = VISITOR_ID_MIN,
        id_max: int = VISITOR_ID_MAX,
        ttl_s: float = 3600.0,
    ) -> None:
        self.pool_file = os.path.expanduser(str(pool_file))
        self.id_min = int(id_min)
        self.id_max = int(id_max)
        self.ttl_s = float(ttl_s)
        self._pool = _SidPool(self.id_min, self.id_max)
        self._last_seen: Dict[int, float] = {}
        self._mint_count = 0
        self._recycle_count = 0
        self.load()

    def load(self) -> None:
        if not os.path.isfile(self.pool_file):
            return
        try:
            with open(self.pool_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        free = data.get("free_visitor_sids", data.get("free_sids", []))
        if isinstance(free, list):
            self._pool.load_free([int(x) for x in free if isinstance(x, (int, float, str))])
        last_seen = data.get("visitor_last_seen", {})
        if isinstance(last_seen, dict):
            for k, v in last_seen.items():
                try:
                    self._last_seen[int(k)] = float(v)
                except Exception:
                    continue

    def save(self) -> None:
        try:
            parent = os.path.dirname(self.pool_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            payload = {
                "free_visitor_sids": self._pool.free_list(),
                "visitor_last_seen": {str(k): float(v) for k, v in self._last_seen.items()},
            }
            tmp = f"{self.pool_file}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, self.pool_file)
        except Exception:
            logger.warning("failed to save visitor pool %s", self.pool_file, exc_info=True)

    def touch(self, sid: int, ts: float) -> None:
        if self.id_min <= int(sid) <= self.id_max:
            self._last_seen[int(sid)] = float(ts)
            self._pool.mark_used(int(sid))

    def alloc(self, ts: float) -> int:
        sid = int(self._pool.alloc())
        self._last_seen[sid] = float(ts)
        self._mint_count += 1
        return sid

    def release(self, sid: int) -> None:
        sid_int = int(sid)
        if not (self.id_min <= sid_int <= self.id_max):
            return
        self._last_seen.pop(sid_int, None)
        self._pool.release(sid_int)
        self._recycle_count += 1

    def recycle_inactive(
        self,
        now_ts: float,
        *,
        active_sids: Set[int],
        ghost_sids: Set[int],
    ) -> List[int]:
        recycled: List[int] = []
        for sid, last in list(self._last_seen.items()):
            if sid in active_sids or sid in ghost_sids:
                continue
            if (float(now_ts) - float(last)) < self.ttl_s:
                continue
            self.release(int(sid))
            recycled.append(int(sid))
        return recycled

    @property
    def mint_count(self) -> int:
        return int(self._mint_count)

    @property
    def recycle_count(self) -> int:
        return int(self._recycle_count)


class ProvisionalPool:
    def __init__(
        self,
        id_min: int = PROVISIONAL_ID_MIN,
        id_max: int = PROVISIONAL_ID_MAX,
    ) -> None:
        self.id_min = int(id_min)
        self.id_max = int(id_max)
        self._pool = _SidPool(self.id_min, self.id_max)

    def alloc(self) -> int:
        return int(self._pool.alloc())

    def release(self, sid: int) -> None:
        self._pool.release(int(sid))

    def is_provisional(self, sid: int) -> bool:
        return self.id_min <= int(sid) <= self.id_max

    def used_sids(self) -> Set[int]:
        return set(int(s) for s in self._pool._used)

    def free_count(self) -> int:
        used = len(self._pool._used)
        span = int(self.id_max) - int(self.id_min) + 1
        return max(0, span - used)

    def used_count(self) -> int:
        return int(len(self._pool._used))

    def reclaim_orphans(self, live_sids: Set[int]) -> List[int]:
        """Release provisionals that are marked used but not in live_sids."""
        live = {int(s) for s in live_sids}
        recycled: List[int] = []
        for sid in list(self._pool._used):
            if int(sid) in live:
                continue
            self.release(int(sid))
            recycled.append(int(sid))
        return recycled


def default_household_paths(root: Optional[Path] = None) -> Dict[str, str]:
    base = root or Path(os.path.expanduser("~/.noesis/household"))
    return {
        "residents_file": str(base / "residents.json"),
        "visitor_pool_file": str(base / "sid_pool.json"),
    }


def copy_gallery_embeddings(
    gallery: Dict[int, Any],
    src_sid: int,
    dst_sid: int,
) -> int:
    """Copy deque embeddings from src to dst; returns count copied."""
    src_entries = gallery.get(int(src_sid))
    if not src_entries:
        return 0
    dst = gallery[int(dst_sid)]
    count = 0
    for ts, emb in list(src_entries):
        if emb is None:
            continue
        dst.append((float(ts), np.asarray(emb, dtype=np.float32)))
        count += 1
    return count
