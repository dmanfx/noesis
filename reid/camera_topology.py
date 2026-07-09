"""Camera topology loader for household identity overlap permits."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverlapPairParams:
    """Parameters for an enabled overlap region between two cameras."""

    camera_a: str
    camera_b: str
    source_a: int
    source_b: int
    max_world_dist_m: float
    max_time_delta_s: float
    require_appearance_sim: float
    enabled: bool = True


class CameraTopology:
    """Maps camera names to source_id and provides overlap-pair lookup."""

    def __init__(self) -> None:
        self._name_to_source: Dict[str, int] = {}
        self._source_to_name: Dict[int, str] = {}
        self._overlap_pairs: Dict[Tuple[int, int], OverlapPairParams] = {}
        self.overlap_allow_appearance_only: bool = False
        self.source_path: Optional[str] = None

    @property
    def cameras(self) -> Dict[str, int]:
        return dict(self._name_to_source)

    def source_id_for_name(self, name: str) -> Optional[int]:
        try:
            return int(self._name_to_source[str(name).strip()])
        except Exception:
            return None

    def name_for_source_id(self, source_id: int) -> Optional[str]:
        return self._source_to_name.get(int(source_id))

    def overlap_params(self, source_a: int, source_b: int) -> Optional[OverlapPairParams]:
        key = _pair_key(int(source_a), int(source_b))
        params = self._overlap_pairs.get(key)
        if params is None or not params.enabled:
            return None
        return params

    def is_overlap_pair(self, source_a: int, source_b: int) -> bool:
        return self.overlap_params(source_a, source_b) is not None


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    ia, ib = int(a), int(b)
    return (ia, ib) if ia <= ib else (ib, ia)


def _resolve_topology_path(path: Optional[str]) -> Optional[str]:
    if path:
        expanded = os.path.expanduser(str(path))
        if os.path.isfile(expanded):
            return expanded
        return None
    env_path = os.environ.get("NOESIS_CAMERA_TOPOLOGY_FILE")
    if env_path:
        expanded = os.path.expanduser(str(env_path))
        if os.path.isfile(expanded):
            return expanded
    default_rel = "config/camera_topology.yaml"
    if os.path.isfile(default_rel):
        return default_rel
    return None


def load_camera_topology(path: Optional[str] = None) -> CameraTopology:
    """Load topology YAML. Missing/invalid file → empty topology (exclusivity only)."""
    topo = CameraTopology()
    resolved = _resolve_topology_path(path)
    if resolved is None:
        if path:
            logger.warning("camera topology file not found: %s (exclusivity without overlap permits)", path)
        return topo

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("failed to load camera topology %s: %s", resolved, exc)
        return topo

    topo.source_path = resolved
    cameras = raw.get("cameras") or {}
    if isinstance(cameras, dict):
        for name, spec in cameras.items():
            if not isinstance(spec, dict):
                continue
            try:
                sid = int(spec.get("source_id"))
            except Exception:
                continue
            cam_name = str(name).strip()
            if not cam_name:
                continue
            topo._name_to_source[cam_name] = sid
            topo._source_to_name[sid] = cam_name

    topo.overlap_allow_appearance_only = bool(raw.get("overlap_allow_appearance_only", False))

    overlaps = raw.get("overlaps") or []
    if isinstance(overlaps, list):
        for entry in overlaps:
            if not isinstance(entry, dict):
                continue
            cam_list = entry.get("cameras") or []
            if not isinstance(cam_list, (list, tuple)) or len(cam_list) != 2:
                continue
            name_a, name_b = str(cam_list[0]).strip(), str(cam_list[1]).strip()
            src_a = topo.source_id_for_name(name_a)
            src_b = topo.source_id_for_name(name_b)
            if src_a is None or src_b is None:
                logger.warning(
                    "overlap pair [%s, %s] skipped: unknown camera name (loaded from %s)",
                    name_a,
                    name_b,
                    resolved,
                )
                continue
            try:
                params = OverlapPairParams(
                    camera_a=name_a,
                    camera_b=name_b,
                    source_a=int(src_a),
                    source_b=int(src_b),
                    max_world_dist_m=float(entry.get("max_world_dist_m", 1.25)),
                    max_time_delta_s=float(entry.get("max_time_delta_s", 0.35)),
                    require_appearance_sim=float(entry.get("require_appearance_sim", 0.60)),
                    enabled=bool(entry.get("enabled", True)),
                )
            except Exception:
                continue
            topo._overlap_pairs[_pair_key(src_a, src_b)] = params

    logger.info(
        "loaded camera topology from %s: cameras=%s overlaps=%d",
        resolved,
        {n: s for n, s in sorted(topo._name_to_source.items())},
        len(topo._overlap_pairs),
    )
    return topo
