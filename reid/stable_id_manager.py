import threading
import time
import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple, Any, Iterable
import heapq
import os
import json

import numpy as np

from .embedding_extractor import EmbeddingExtractor


BBox = Tuple[float, float, float, float]  # left, top, width, height


class StableIDManager:
    """Assigns stable, global IDs across cameras using visual embeddings.

    - Maintains mapping (sensor_id, ds_obj_id) -> stable_id for active tracks
    - Keeps a ghost registry per camera for re-association after disappearance
    - Maintains an identity gallery (stable_id -> recent embeddings) for cross-camera matches
    - Optionally allows the same stable_id to be active in multiple cameras (overlapping FoVs)
    """

    _POSE_FEATURE_KEYS = (
        "height_proxy_norm",
        "torso_len_norm",
        "leg_len_norm",
        "torso_leg_ratio",
        "leg_height_ratio",
        "shoulder_width_norm",
        "hip_width_norm",
        "left_upper_arm_norm",
        "left_lower_arm_norm",
        "right_upper_arm_norm",
        "right_lower_arm_norm",
        "left_upper_leg_norm",
        "left_lower_leg_norm",
        "right_upper_leg_norm",
        "right_lower_leg_norm",
        "left_arm_ratio",
        "right_arm_ratio",
        "left_leg_ratio",
        "right_leg_ratio",
        "arm_symmetry",
        "leg_symmetry",
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        model_name: str = "osnet_x1_0",
        image_size: Tuple[int, int] = (256, 128),
        embed_interval_s: float = 1.0,
        max_ghost_age_s: float = 60.0,
        ghost_queue_max: int = 200,
        gallery_size: int = 20,
        cos_sim_threshold: float = 0.62,
        cos_sim_high_threshold: float = 0.72,
        allow_multi_zone_active: bool = True,
        # Robustness/appearance tuning
        crop_expand: float = 0.12,
        tta_flip: bool = True,
        min_crop_h: int = 64,
        min_laplacian_var: float = 12.0,
        adaptive_penalty: bool = True,
        size_penalty_alpha: float = 0.08,
        brightness_penalty_beta: float = 0.05,
        # Additional disambiguation
        spatial_penalty: bool = True,
        spatial_penalty_delta: float = 0.06,
        color_penalty_gamma: float = 0.07,
        # Partial feature fusion and smoothing
        stripe_fusion: bool = True,
        stripe_count: int = 3,
        multi_scale_crops: bool = True,
        ema_alpha: float = 0.20,
        # Anti-merge guard
        active_id_guard_strict: bool = True,
        active_id_guard_margin: float = 0.03,
        # Ghost strictness
        ghost_strict_age_s: float = 8.0,
        ghost_extra_margin: float = 0.01,
        # Soft cap on active IDs
        max_active_ids_per_sensor: int = 6,
        new_id_confirm_frames_at_cap: int = 2,
        active_evict_grace_s: float = 10.0,
        # Cross-camera handoff
        xcam_handoff_window_s: float = 4.0,
        xcam_handoff_margin: float = 0.02,
        # Global ID pool soft-cap
        max_total_ids: int = 12,
        total_id_reuse: bool = True,
        total_id_reuse_min_age_s: float = 600.0,
        # New-ID hysteresis (global, even when not at cap)
        new_id_hysteresis_frames: int = 2,
        # SID allocator persistence (smarter restart)
        sid_pool_file: str = "~/.noesis/sid_pool.json",
        # External embedding support (e.g., DeepStream SGIE tensor outputs).
        # When disabled, embeddings are extracted internally from BGR crops.
        use_extractor: bool = True,
        # StableID similarity compute backend (hybrid CPU/GPU mode).
        compute_backend: str = "auto",
        gpu_device: Optional[str] = None,
        gpu_min_gallery: int = 32,
        # Pose feature support (optional).
        pose_enabled: bool = False,
        pose_weight: float = 0.15,
        pose_sim_threshold: float = 0.55,
        pose_sim_high_threshold: float = 0.65,
        pose_only_threshold: float = 0.80,
        pose_min_valid_frac: float = 0.45,
        pose_min_mean_conf: float = 0.50,
        pose_min_features: int = 6,
        pose_interval_s: float = 0.75,
        pose_gallery_size: int = 8,
        pose_max_age_s: float = 30.0,
        pose_max_total_entries: int = 0,
        # Alias/merge controls (default off for backward compat)
        aliases_enabled: bool = False,
        alias_file: str = "~/.noesis/reid_aliases.json",
        alias_autosave: bool = True,
        alias_append_default: bool = True,
        copresence_window_s: float = 600.0,
        min_embeddings_for_suggest: int = 3,
        suggest_min_sim: float = 0.92,
        suggest_mnn_margin: float = 0.02,
        suggest_pose_sim_low: float = 0.70,
        suggest_pose_sim_high: float = 0.90,
        alias_history_max: int = 1000,
    ) -> None:
        self._lock = threading.RLock()
        self._use_extractor = bool(use_extractor)
        self.compute_backend = str(compute_backend or "auto").strip().lower()
        self.gpu_device = str(gpu_device or device or "cuda:0")
        self.gpu_min_gallery = int(max(1, gpu_min_gallery))
        self._backend_mode = "cpu"
        self._backend_last_error: Optional[str] = None
        self._match_latency_ms: Deque[float] = deque(maxlen=512)
        self._torch = None
        self._torch_device = None
        self._gpu_enabled = False
        self.extractor: Optional[EmbeddingExtractor]
        if self._use_extractor:
            self.extractor = EmbeddingExtractor(
                model_path=model_path,
                device=device,
                image_size=image_size,
                model_name=model_name,
            )
        else:
            self.extractor = None

        self.embed_interval_s = float(embed_interval_s)
        self.max_ghost_age_s = float(max_ghost_age_s)
        self.cos_sim_threshold = float(cos_sim_threshold)
        self.cos_sim_high_threshold = float(cos_sim_high_threshold)
        self.allow_multi_zone_active = bool(allow_multi_zone_active)
        self.crop_expand = float(crop_expand)
        self.tta_flip = bool(tta_flip)
        self.min_crop_h = int(min_crop_h)
        self.min_laplacian_var = float(min_laplacian_var)
        self.adaptive_penalty = bool(adaptive_penalty)
        self.size_penalty_alpha = float(size_penalty_alpha)
        self.brightness_penalty_beta = float(brightness_penalty_beta)
        self.spatial_penalty = bool(spatial_penalty)
        self.spatial_penalty_delta = float(spatial_penalty_delta)
        self.color_penalty_gamma = float(color_penalty_gamma)
        self.stripe_fusion = bool(stripe_fusion)
        self.stripe_count = int(max(1, stripe_count))
        self.multi_scale_crops = bool(multi_scale_crops)
        self.ema_alpha = float(ema_alpha)
        self.active_id_guard_strict = bool(active_id_guard_strict)
        self.active_id_guard_margin = float(active_id_guard_margin)
        self.ghost_strict_age_s = float(ghost_strict_age_s)
        self.ghost_extra_margin = float(ghost_extra_margin)
        self.max_active_ids_per_sensor = int(max_active_ids_per_sensor)
        self.new_id_confirm_frames_at_cap = int(new_id_confirm_frames_at_cap)
        self.active_evict_grace_s = float(active_evict_grace_s)
        self.xcam_handoff_window_s = float(xcam_handoff_window_s)
        self.xcam_handoff_margin = float(xcam_handoff_margin)
        self.max_total_ids = int(max_total_ids)
        self.total_id_reuse = bool(total_id_reuse)
        self.total_id_reuse_min_age_s = float(total_id_reuse_min_age_s)
        self.new_id_hysteresis_frames = int(new_id_hysteresis_frames)
        self.sid_pool_file = os.path.expanduser(str(sid_pool_file))
        self.pose_enabled = bool(pose_enabled)
        self.pose_weight = float(pose_weight)
        self.pose_sim_threshold = float(pose_sim_threshold)
        self.pose_sim_high_threshold = float(pose_sim_high_threshold)
        self.pose_only_threshold = float(pose_only_threshold)
        self.pose_min_valid_frac = float(pose_min_valid_frac)
        self.pose_min_mean_conf = float(pose_min_mean_conf)
        self.pose_min_features = int(max(0, pose_min_features))
        self.pose_interval_s = float(pose_interval_s)
        self.pose_gallery_size = int(max(1, pose_gallery_size))
        self.pose_max_age_s = float(pose_max_age_s)
        self.pose_max_total_entries = int(pose_max_total_entries)
        if self.pose_max_total_entries <= 0:
            base = max(1, int(self.max_total_ids))
            self.pose_max_total_entries = max(self.pose_gallery_size, int(base * self.pose_gallery_size))
        self.gallery_size = int(max(1, gallery_size))
        self.aliases_enabled = bool(aliases_enabled)
        self.alias_file = os.path.expanduser(str(alias_file))
        self.alias_autosave = bool(alias_autosave)
        self.alias_append_default = bool(alias_append_default)
        self.copresence_window_s = float(copresence_window_s)
        self.min_embeddings_for_suggest = int(max(1, min_embeddings_for_suggest))
        self.suggest_min_sim = float(suggest_min_sim)
        self.suggest_mnn_margin = float(suggest_mnn_margin)
        self.suggest_pose_sim_low = float(suggest_pose_sim_low)
        self.suggest_pose_sim_high = float(suggest_pose_sim_high)
        self.alias_history_max = int(max(1, alias_history_max))

        # Active tracks: (sensor_id, ds_obj_id) -> record
        self.active_tracks: Dict[Tuple[int, int], Dict] = {}

        # Ghosts by camera: sensor_id -> deque of ghost records
        self.ghosts: Dict[int, Deque[Dict]] = defaultdict(lambda: deque(maxlen=ghost_queue_max))

        # Identity gallery: stable_id -> deque of (ts, embedding)
        self.gallery: Dict[int, Deque[Tuple[float, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self.gallery_size)
        )
        # EMA centroid per stable id
        self.sid_centroid: Dict[int, np.ndarray] = {}
        # Pose feature gallery: stable_id -> deque of (ts, pose_vec)
        self.pose_gallery: Dict[int, Deque[Tuple[float, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=self.pose_gallery_size)
        )
        # Pose centroid per stable id
        self.pose_centroid: Dict[int, np.ndarray] = {}
        # Pose last-seen timestamps per stable id
        self.pose_last_seen: Dict[int, float] = {}

        # Where each identity is currently active: stable_id -> set of (sensor_id, zone)
        self.active_zones: Dict[int, set] = defaultdict(set)

        # Last known bbox/appearance per stable_id (for adaptive penalties)
        self.sid_last_bbox: Dict[int, BBox] = {}
        self.sid_last_brightness: Dict[int, float] = {}
        self.sid_last_color: Dict[int, np.ndarray] = {}
        # Global last seen timestamp per stable_id (any sensor)
        self.sid_global_last_seen: Dict[int, float] = {}
        # Global first seen timestamp per stable_id
        self.sid_global_first_seen: Dict[int, float] = {}
        # Alias map: src_sid -> dst_sid (may chain)
        self.sid_alias: Dict[int, int] = {}
        # All IDs participating in any alias (src or dst)
        self.sid_alias_reserved: set[int] = set()
        # Alias audit trail
        self.alias_history: List[Dict[str, Any]] = []
        # Copresence map (low_sid, high_sid) -> last timestamp
        self.sid_last_copresent: Dict[Tuple[int, int], float] = {}
        # Pending new-ID confirmation counters at cap
        self._pending_new_counts: Dict[Tuple[int, int], int] = {}
        self._pending_new_ts: Dict[Tuple[int, int], float] = {}
        # Pending stable IDs for new tracks (avoid negative provisional IDs).
        self._pending_new_sids: Dict[Tuple[int, int], int] = {}

        self.next_stable_id = 1
        # Free-list allocator state
        self._free_sids: List[int] = []
        self._free_sids_set: set[int] = set()
        if self.aliases_enabled:
            self._load_aliases()
            self._refresh_alias_reserved()
        self._load_sid_pool()
        if self.aliases_enabled and self.sid_alias_reserved:
            self._free_sids = [sid for sid in self._free_sids if sid not in self.sid_alias_reserved]
            heapq.heapify(self._free_sids)
            self._free_sids_set = set(self._free_sids)
        self._init_compute_backend()

    def _init_compute_backend(self) -> None:
        pref = str(self.compute_backend or "auto").strip().lower()
        strict_gpu = pref in ("gpu", "cuda", "torch")
        if pref in ("cpu", "numpy"):
            self._backend_mode = "cpu"
            return
        try:
            import torch  # type: ignore
        except Exception as exc:
            if strict_gpu:
                raise RuntimeError(f"StableID GPU backend requested but torch import failed: {type(exc).__name__}") from exc
            self._backend_mode = "cpu"
            self._backend_last_error = f"torch_import:{type(exc).__name__}"
            return
        if not bool(getattr(torch, "cuda", None)) or not bool(torch.cuda.is_available()):
            if strict_gpu:
                raise RuntimeError("StableID GPU backend requested but CUDA is unavailable")
            self._backend_mode = "cpu"
            self._backend_last_error = "cuda_unavailable"
            return
        try:
            dev = torch.device(self.gpu_device)
            if getattr(dev, "type", "") != "cuda":
                raise ValueError("non_cuda_device")
            _ = torch.tensor([1.0], device=dev)
        except Exception as exc:
            if strict_gpu:
                raise RuntimeError(f"StableID GPU backend requested but device init failed: {type(exc).__name__}") from exc
            self._backend_mode = "cpu"
            self._backend_last_error = f"cuda_device:{type(exc).__name__}"
            return
        self._torch = torch
        self._torch_device = dev
        self._backend_mode = "gpu"
        self._gpu_enabled = True

    def _record_match_latency(self, start_ns: int) -> None:
        try:
            elapsed_ms = max(0.0, (time.perf_counter_ns() - int(start_ns)) / 1_000_000.0)
            self._match_latency_ms.append(float(elapsed_ms))
        except Exception:
            pass

    @staticmethod
    def _percentile(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 1:
            return None
        return float(np.percentile(arr, q))

    def _similarity_for_candidates(self, emb: np.ndarray, candidates: List[int]) -> Dict[int, float]:
        sims: Dict[int, float] = {}
        if emb is None or len(candidates) < 1:
            return sims
        emb_vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        centroid_rows: List[np.ndarray] = []
        sid_rows: List[int] = []
        for sid in candidates:
            sid_int = int(sid)
            centroid = self.sid_centroid.get(sid_int)
            if centroid is None:
                vecs = self.gallery.get(sid_int)
                if vecs:
                    embs = [v for (_ts, v) in vecs if v is not None]
                    if embs:
                        centroid = np.mean(np.stack(embs, axis=0), axis=0)
                        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                        centroid = centroid.astype(np.float32)
                        self.sid_centroid[sid_int] = centroid
            if centroid is None:
                continue
            c = np.asarray(centroid, dtype=np.float32).reshape(-1)
            if c.shape[0] != emb_vec.shape[0]:
                continue
            centroid_rows.append(c)
            sid_rows.append(sid_int)
        if not sid_rows:
            return sims

        use_gpu = bool(self._gpu_enabled and self._backend_mode == "gpu" and len(sid_rows) >= int(self.gpu_min_gallery))
        if use_gpu:
            try:
                torch = self._torch
                if torch is None or self._torch_device is None:
                    raise RuntimeError("torch_backend_uninitialized")
                q = torch.as_tensor(emb_vec, dtype=torch.float32, device=self._torch_device)
                q = q / (torch.norm(q) + 1e-12)
                mat = torch.as_tensor(np.stack(centroid_rows, axis=0), dtype=torch.float32, device=self._torch_device)
                mat = mat / (torch.linalg.norm(mat, dim=1, keepdim=True) + 1e-12)
                sim_vec = torch.matmul(mat, q)
                sim_np = sim_vec.detach().cpu().numpy().astype(np.float32, copy=False)
                for idx, sid_int in enumerate(sid_rows):
                    sims[int(sid_int)] = float(sim_np[idx])
                return sims
            except Exception as exc:
                self._backend_last_error = f"gpu_similarity:{type(exc).__name__}"
                raise RuntimeError(f"StableID GPU similarity failed: {type(exc).__name__}") from exc

        mat_np = np.stack(centroid_rows, axis=0).astype(np.float32, copy=False)
        sim_np = np.matmul(mat_np, emb_vec.reshape(-1, 1)).reshape(-1)
        for idx, sid_int in enumerate(sid_rows):
            sims[int(sid_int)] = float(sim_np[idx])
        return sims

    # --------------- Allocator -----------------
    def _alloc_sid(self) -> int:
        while self._free_sids:
            sid = heapq.heappop(self._free_sids)
            self._free_sids_set.discard(sid)
            if not self._is_alias_reserved(int(sid)):
                self.sid_global_first_seen.pop(int(sid), None)
                return int(sid)
        sid = int(self.next_stable_id)
        self.next_stable_id += 1
        return int(sid)

    def _free_sid(self, sid: int) -> None:
        try:
            sid = int(sid)
        except Exception:
            return
        if sid <= 0:
            return
        if self._is_alias_reserved(int(sid)):
            return
        if sid in self._free_sids_set:
            return
        heapq.heappush(self._free_sids, sid)
        self._free_sids_set.add(sid)
        self._save_sid_pool()

    def _purge_sid_state(self, sid: int) -> None:
        try:
            sid = int(sid)
        except Exception:
            return
        try:
            self.gallery.pop(sid, None)
            self.sid_centroid.pop(sid, None)
            self.sid_last_bbox.pop(sid, None)
            self.sid_last_brightness.pop(sid, None)
            self.sid_last_color.pop(sid, None)
            self.active_zones.pop(sid, None)
            self.pose_gallery.pop(sid, None)
            self.pose_centroid.pop(sid, None)
            self.pose_last_seen.pop(sid, None)
            self.sid_global_last_seen.pop(sid, None)
            self.sid_global_first_seen.pop(sid, None)
        except Exception:
            pass

    def _load_sid_pool(self) -> None:
        try:
            if not os.path.exists(self.sid_pool_file):
                return
            with open(self.sid_pool_file, 'r') as f:
                data = json.load(f)
            pool = data.get('free_sids', []) if isinstance(data, dict) else []
            pool = [int(x) for x in pool if isinstance(x, int) and x > 0]
            pool = sorted(set(pool))[:32]
            for sid in pool:
                if sid not in self._free_sids_set:
                    heapq.heappush(self._free_sids, sid)
                    self._free_sids_set.add(sid)
        except Exception:
            pass

    def _save_sid_pool(self) -> None:
        try:
            pool = sorted(list(self._free_sids_set))[:32]
            os.makedirs(os.path.dirname(self.sid_pool_file), exist_ok=True)
            with open(self.sid_pool_file, 'w') as f:
                json.dump(
                    {
                        'free_sids': pool,
                    },
                    f,
                )
        except Exception:
            pass

    def _load_aliases(self) -> None:
        if not self.aliases_enabled:
            return
        try:
            if not self.alias_file or not os.path.exists(self.alias_file):
                return
            with open(self.alias_file, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return
            raw_aliases = data.get("aliases", {})
            if isinstance(raw_aliases, dict):
                for k, v in raw_aliases.items():
                    try:
                        src = int(k)
                        dst = int(v)
                    except Exception:
                        continue
                    if src <= 0 or dst <= 0 or src == dst:
                        continue
                    self.sid_alias[src] = dst
            hist_raw = data.get("history", [])
            if isinstance(hist_raw, list):
                for entry in hist_raw[-self.alias_history_max :]:
                    if isinstance(entry, dict):
                        self.alias_history.append(entry)
            # Normalize alias map (avoid loops / self-maps).
            for src in list(self.sid_alias.keys()):
                canon = self.canonical_sid(src)
                if canon == src:
                    self.sid_alias.pop(src, None)
                else:
                    self.sid_alias[src] = canon
        except Exception:
            pass

    def _save_aliases(self) -> None:
        if not (self.aliases_enabled and self.alias_autosave):
            return
        try:
            if not self.alias_file:
                return
            # Prune history before save.
            if len(self.alias_history) > self.alias_history_max:
                self.alias_history = self.alias_history[-self.alias_history_max :]
            payload = {
                "version": 1,
                "aliases": {str(k): int(v) for k, v in self.sid_alias.items()},
                "history": list(self.alias_history),
            }
            dir_path = os.path.dirname(self.alias_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            tmp_path = f"{self.alias_file}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.alias_file)
        except Exception:
            pass

    # --------------- Alias helpers -----------------
    def canonical_sid(self, sid: int) -> int:
        """Return the canonical SID after following any alias chain."""
        try:
            sid_int = int(sid)
        except Exception:
            return int(sid) if isinstance(sid, int) else 0
        if sid_int <= 0:
            return sid_int
        visited: List[int] = []
        curr = sid_int
        for _ in range(64):
            nxt = self.sid_alias.get(curr)
            if nxt is None or int(nxt) == curr:
                break
            try:
                nxt_int = int(nxt)
            except Exception:
                break
            if nxt_int in visited:
                break
            visited.append(curr)
            curr = nxt_int
        for v in visited:
            self.sid_alias[v] = curr
        return int(curr)

    def _is_alias_src(self, sid: int) -> bool:
        try:
            sid_int = int(sid)
        except Exception:
            return False
        return self.canonical_sid(sid_int) != sid_int

    def _refresh_alias_reserved(self) -> None:
        self.sid_alias_reserved = set(self.sid_alias.keys()) | set(self.sid_alias.values())

    def _is_alias_reserved(self, sid: int) -> bool:
        try:
            return int(sid) in self.sid_alias_reserved
        except Exception:
            return False

    def _record_alias_event(
        self,
        action: str,
        src_sid: int,
        dst_sid: Optional[int],
        ts: Optional[float] = None,
        *,
        reason: Optional[str] = None,
        sim: Optional[float] = None,
        pose_sim: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            entry: Dict[str, Any] = {
                "ts": float(ts if ts is not None else time.time()),
                "action": str(action),
                "src": int(src_sid),
            }
            if dst_sid is not None:
                entry["dst"] = int(dst_sid)
            if reason:
                entry["reason"] = str(reason)
            if sim is not None and math.isfinite(float(sim)):
                entry["sim"] = float(sim)
            if pose_sim is not None and math.isfinite(float(pose_sim)):
                entry["pose_sim"] = float(pose_sim)
            if extra:
                for k, v in extra.items():
                    if v is None:
                        continue
                    if isinstance(v, (int, float, str, bool)):
                        entry[str(k)] = v
            self.alias_history.append(entry)
        except Exception:
            pass

    def _gallery_vectors(self, sid: int) -> List[np.ndarray]:
        dq = self.gallery.get(int(sid))
        if not dq:
            return []
        return [vec for (_ts, vec) in dq if vec is not None]

    def _recompute_sid_centroid(self, sid: int) -> Optional[np.ndarray]:
        sid_int = int(sid)
        vecs = self._gallery_vectors(sid_int)
        if not vecs:
            self.sid_centroid.pop(sid_int, None)
            return None
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroid = centroid.astype(np.float32)
        self.sid_centroid[sid_int] = centroid
        return centroid

    def _pose_centroid_for_sid(self, sid: int) -> Optional[np.ndarray]:
        sid_int = int(sid)
        centroid = self.pose_centroid.get(sid_int)
        if centroid is not None:
            return centroid
        dq = self.pose_gallery.get(sid_int)
        if not dq:
            return None
        vecs = [v for (_ts, v) in dq if v is not None]
        if not vecs:
            return None
        centroid = np.mean(np.stack(vecs, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        centroid = centroid.astype(np.float32)
        self.pose_centroid[sid_int] = centroid
        return centroid

    def _pose_similarity_between_sids(self, sid_a: int, sid_b: int) -> Optional[float]:
        a = self._pose_centroid_for_sid(sid_a)
        b = self._pose_centroid_for_sid(sid_b)
        if a is None or b is None:
            return None
        return self._cosine(a, b)

    def observe_copresence(self, sids: List[int], ts: float) -> None:
        if not self.aliases_enabled:
            return
        if self.copresence_window_s <= 0.0:
            return
        try:
            canon = {self.canonical_sid(sid) for sid in sids if int(sid) > 0}
        except Exception:
            canon = set()
        canon = {sid for sid in canon if sid > 0}
        if len(canon) < 2:
            return
        ordered = sorted(canon)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a = ordered[i]
                b = ordered[j]
                self.sid_last_copresent[(a, b)] = float(ts)
        self._prune_copresence(float(ts))

    def _prune_copresence(self, now_ts: float) -> None:
        if self.copresence_window_s <= 0.0:
            return
        cutoff = float(now_ts) - float(self.copresence_window_s) * 2.0
        for key, last in list(self.sid_last_copresent.items()):
            try:
                if float(last) < cutoff:
                    self.sid_last_copresent.pop(key, None)
            except Exception:
                self.sid_last_copresent.pop(key, None)

    def was_copresent_recently(self, a: int, b: int, now_ts: float) -> bool:
        if self.copresence_window_s <= 0.0:
            return False
        a_can = self.canonical_sid(a)
        b_can = self.canonical_sid(b)
        if a_can <= 0 or b_can <= 0 or a_can == b_can:
            return False
        key = (a_can, b_can) if a_can < b_can else (b_can, a_can)
        last = self.sid_last_copresent.get(key)
        if last is None:
            return False
        return float(last) > float(now_ts) - float(self.copresence_window_s)

    # --------------- Utility -----------------
    @staticmethod
    def _bbox_to_int(b: BBox) -> Tuple[int, int, int, int]:
        x, y, w, h = b
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _crop(frame_bgr: Optional[np.ndarray], bbox: BBox, expand: float = 0.0) -> Optional[np.ndarray]:
        if frame_bgr is None:
            return None
        H, W = frame_bgr.shape[:2]
        x, y, w, h = StableIDManager._bbox_to_int(bbox)
        if expand > 0:
            # Expand symmetrically and clamp
            ex = int(w * expand)
            ey = int(h * expand)
            x = max(0, x - ex)
            y = max(0, y - ey)
            w = min(W - x, w + 2 * ex)
            h = min(H - y, h + 2 * ey)
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        return frame_bgr[y : y + h, x : x + w].copy()

    @staticmethod
    def _crop_brightness(crop_bgr: np.ndarray) -> float:
        try:
            import cv2
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            return float(np.mean(hsv[..., 2])) / 255.0
        except Exception:
            return 0.5

    @staticmethod
    def _crop_blur_var(crop_bgr: np.ndarray) -> float:
        try:
            import cv2
            g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(g, cv2.CV_64F).var())
        except Exception:
            return 1e9  # if cv2 absent, do not block updates

    @staticmethod
    def _crop_color_hist(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            import cv2
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
            vec = np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)
            vec = vec / (np.sum(vec) + 1e-6)
            return vec
        except Exception:
            return None

    def _extract_embedding_from_crop(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        try:
            import cv2
            crops: List[np.ndarray] = []

            H, W = crop_bgr.shape[:2]
            # Always include full crop
            crops.append(crop_bgr)

            # Multi-scale halves
            if self.multi_scale_crops:
                mid_y = H // 2
                top = crop_bgr[0:mid_y, :]
                bot = crop_bgr[mid_y:H, :]
                if top.shape[0] >= self.min_crop_h:
                    crops.append(top)
                if bot.shape[0] >= self.min_crop_h:
                    crops.append(bot)

            # Stripe fusion (vertical stripes)
            if self.stripe_fusion and self.stripe_count >= 2:
                stripe_h = H // self.stripe_count
                for i in range(self.stripe_count):
                    y0 = i * stripe_h
                    y1 = H if i == self.stripe_count - 1 else (i + 1) * stripe_h
                    if (y1 - y0) >= self.min_crop_h:
                        crops.append(crop_bgr[y0:y1, :])

            # Build batch of crops (with optional flips for TTA)
            batch: List[np.ndarray] = []
            if self.tta_flip:
                for c in crops:
                    batch.append(c)
                    batch.append(cv2.flip(c, 1))
            else:
                batch = crops

            if self.extractor is None:
                return None
            feats = self.extractor.extract(batch)
            if feats is None or feats.shape[0] < 1:
                return None
            vec = np.mean(feats.astype(np.float32), axis=0)
            n = np.linalg.norm(vec) + 1e-12
            return (vec / n).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        # Both are assumed L2-normalized
        return float(np.dot(a, b))

    def _pose_quality_ok(self, pose_quality: Optional[Dict[str, float]]) -> bool:
        if not self.pose_enabled:
            return False
        if pose_quality is None:
            return False
        mean_conf = pose_quality.get("kpt_mean_conf")
        if mean_conf is None:
            mean_conf = pose_quality.get("mean_conf")
        valid_frac = pose_quality.get("kpt_valid_frac")
        if valid_frac is None:
            valid_frac = pose_quality.get("valid_frac")
        try:
            mean_conf_f = float(mean_conf) if mean_conf is not None else 0.0
        except Exception:
            mean_conf_f = 0.0
        try:
            valid_frac_f = float(valid_frac) if valid_frac is not None else 0.0
        except Exception:
            valid_frac_f = 0.0
        if not math.isfinite(mean_conf_f) or not math.isfinite(valid_frac_f):
            return False
        if mean_conf_f < float(self.pose_min_mean_conf):
            return False
        if valid_frac_f < float(self.pose_min_valid_frac):
            return False
        return True

    def _pose_vector_from_features(self, pose_features: Dict[str, float]) -> Optional[np.ndarray]:
        if not pose_features:
            return None
        values: List[float] = []
        present = 0
        for key in self._POSE_FEATURE_KEYS:
            if key in pose_features:
                try:
                    val = float(pose_features.get(key))
                except Exception:
                    val = None
                if val is not None and math.isfinite(val):
                    values.append(float(val))
                    present += 1
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
        if present < int(self.pose_min_features):
            return None
        vec = np.asarray(values, dtype=np.float32)
        n = float(np.linalg.norm(vec) + 1e-12)
        return (vec / n).astype(np.float32)

    def _pose_similarity(self, pose_vec: np.ndarray, sid: int) -> Optional[float]:
        if pose_vec is None:
            return None
        sid_int = int(sid)
        centroid = self.pose_centroid.get(sid_int)
        if centroid is None:
            dq = self.pose_gallery.get(sid_int)
            if not dq:
                return None
            try:
                vecs = [v for (_ts, v) in dq if v is not None]
                if not vecs:
                    return None
                centroid = np.mean(np.stack(vecs, axis=0), axis=0)
                norm = np.linalg.norm(centroid) + 1e-12
                centroid = centroid / norm
            except Exception:
                return None
        return self._cosine(pose_vec, centroid)

    def _update_pose_state(self, sid: int, pose_vec: np.ndarray, ts: float) -> None:
        sid_int = int(sid)
        dq = self.pose_gallery[sid_int]
        dq.append((float(ts), pose_vec))
        self.pose_last_seen[sid_int] = float(ts)
        try:
            old = self.pose_centroid.get(sid_int)
            if old is None:
                newc = pose_vec
            else:
                newc = (1.0 - self.ema_alpha) * old + self.ema_alpha * pose_vec
            newc = newc / (np.linalg.norm(newc) + 1e-12)
            self.pose_centroid[sid_int] = newc.astype(np.float32)
        except Exception:
            pass

    def _prune_pose_gallery(self, now_ts: float) -> None:
        if not self.pose_enabled:
            return
        t = float(now_ts)
        max_age = float(self.pose_max_age_s)
        for sid, dq in list(self.pose_gallery.items()):
            removed = False
            if max_age > 0.0:
                while dq and (t - float(dq[0][0])) > max_age:
                    dq.popleft()
                    removed = True
            if not dq:
                self.pose_gallery.pop(sid, None)
                self.pose_centroid.pop(sid, None)
                self.pose_last_seen.pop(sid, None)
                continue
            if removed:
                self.pose_centroid.pop(sid, None)
        max_entries = int(self.pose_max_total_entries)
        if max_entries <= 0:
            return
        total = sum(len(dq) for dq in self.pose_gallery.values())
        if total <= max_entries:
            return
        heap: List[Tuple[float, int]] = []
        for sid, dq in self.pose_gallery.items():
            if dq:
                heapq.heappush(heap, (float(dq[0][0]), int(sid)))
        while total > max_entries and heap:
            ts0, sid = heapq.heappop(heap)
            dq = self.pose_gallery.get(int(sid))
            if not dq:
                continue
            if dq and float(dq[0][0]) != float(ts0):
                heapq.heappush(heap, (float(dq[0][0]), int(sid)))
                continue
            dq.popleft()
            total -= 1
            if dq:
                heapq.heappush(heap, (float(dq[0][0]), int(sid)))
                self.pose_centroid.pop(int(sid), None)
            else:
                self.pose_gallery.pop(int(sid), None)
                self.pose_centroid.pop(int(sid), None)
                self.pose_last_seen.pop(int(sid), None)

    def _pose_gallery_best(
        self,
        pose_vec: np.ndarray,
        now_ts: float,
    ) -> Tuple[Optional[int], float, float]:
        best_id = None
        best_score = -1.0
        best_req = float(self.pose_only_threshold)
        for sid in list(self.pose_gallery.keys()):
            pose_sim = self._pose_similarity(pose_vec, sid)
            if pose_sim is None:
                continue
            req = float(self.pose_only_threshold)
            last_glob = self.sid_global_last_seen.get(int(sid))
            if last_glob is not None and (float(now_ts) - float(last_glob)) <= self.xcam_handoff_window_s:
                req = max(0.0, req - self.xcam_handoff_margin)
            if pose_sim < req:
                continue
            if pose_sim > best_score:
                best_id = int(sid)
                best_score = float(pose_sim)
                best_req = float(req)
        return best_id, float(best_score), float(best_req)

    def _gallery_best(
        self,
        emb: np.ndarray,
        sensor_id: Optional[int] = None,
        curr_bbox: Optional[BBox] = None,
        curr_brightness: Optional[float] = None,
        curr_color: Optional[np.ndarray] = None,
        *,
        pose_vec: Optional[np.ndarray] = None,
        pose_valid: bool = False,
        now_ts: Optional[float] = None,
        min_reid: Optional[float] = None,
    ) -> Tuple[Optional[int], float, float]:
        start_ns = time.perf_counter_ns()
        best_id, best_score = None, -1.0
        best_reid = -1.0
        best_req = float(self.cos_sim_high_threshold)
        candidates: List[int] = []
        seen: set[int] = set()
        for sid in list(self.gallery.keys()):
            try:
                sid_int = int(sid)
            except Exception:
                continue
            sid_can = self.canonical_sid(sid_int) if self.aliases_enabled else sid_int
            if sid_can in seen:
                continue
            seen.add(sid_can)
            candidates.append(sid_can)
        sim_by_sid = self._similarity_for_candidates(emb, candidates)
        for sid in candidates:
            sim = sim_by_sid.get(int(sid))
            if sim is None:
                continue

            score = sim
            if self.adaptive_penalty and curr_bbox is not None:
                # Penalize large scale/brightness shifts relative to last seen for this sid
                last_bbox = self.sid_last_bbox.get(int(sid))
                if last_bbox is not None:
                    _, _, w1, h1 = curr_bbox
                    _, _, w0, h0 = last_bbox
                    a0 = max(1.0, float(w0 * h0))
                    a1 = max(1.0, float(w1 * h1))
                    scale_ratio = a1 / a0 if a1 >= a0 else a0 / a1
                    # scale_ratio >= 1; penalty grows with deviation from 1
                    score -= self.size_penalty_alpha * (scale_ratio - 1.0)
                if curr_brightness is not None:
                    b0 = self.sid_last_brightness.get(int(sid))
                    if b0 is not None:
                        score -= self.brightness_penalty_beta * abs(float(curr_brightness) - float(b0))

            # Spatial penalty when candidate ID already active on this sensor and far away
            if self.spatial_penalty and sensor_id is not None and curr_bbox is not None:
                active_pairs = self.active_zones.get(int(sid), set())
                active_here = any(int(s) == int(sensor_id) for (s, _z) in active_pairs)
                if active_here:
                    last_bbox = self.sid_last_bbox.get(int(sid))
                    if last_bbox is not None:
                        cx = float(curr_bbox[0] + curr_bbox[2] / 2.0)
                        cy = float(curr_bbox[1] + curr_bbox[3] / 2.0)
                        lx = float(last_bbox[0] + last_bbox[2] / 2.0)
                        ly = float(last_bbox[1] + last_bbox[3] / 2.0)
                        dx = cx - lx
                        dy = cy - ly
                        dist = (dx * dx + dy * dy) ** 0.5
                        diag = float((curr_bbox[2] ** 2 + curr_bbox[3] ** 2) ** 0.5)
                        if diag > 0:
                            ratio = max(0.0, dist / diag - 1.0)  # zero until more than one bbox-diag apart
                            score -= self.spatial_penalty_delta * ratio

            # Color penalty for mismatch with last color hist
            if self.color_penalty_gamma > 0.0 and curr_color is not None:
                last_col = self.sid_last_color.get(int(sid))
                if last_col is not None:
                    denom = (np.linalg.norm(curr_color) + 1e-12) * (np.linalg.norm(last_col) + 1e-12)
                    cos = float(np.dot(curr_color, last_col) / denom)
                    cos = max(0.0, min(1.0, cos))
                    color_dist = 1.0 - cos
                    score -= self.color_penalty_gamma * color_dist

            # Candidate-specific cross-camera requirement
            if min_reid is None:
                req = float(self.cos_sim_high_threshold)
                if now_ts is not None:
                    last_glob = self.sid_global_last_seen.get(int(sid))
                    if last_glob is not None and (float(now_ts) - float(last_glob)) <= self.xcam_handoff_window_s:
                        req = max(0.0, req - self.xcam_handoff_margin)
            else:
                req = float(min_reid)
            if score < req:
                continue

            combined = score
            if pose_valid and pose_vec is not None:
                pose_sim = self._pose_similarity(pose_vec, sid)
                if pose_sim is not None and pose_sim >= float(self.pose_sim_high_threshold):
                    combined = score + float(self.pose_weight) * float(pose_sim)

            if combined > best_score:
                best_id = sid
                best_score = float(combined)
                best_reid = float(score)
                best_req = float(req)
        self._record_match_latency(start_ns)
        return best_id, float(best_reid), float(best_req)

    # --------------- Public API --------------
    def list_aliases(self) -> Dict[int, int]:
        with self._lock:
            if not self.sid_alias:
                return {}
            out: Dict[int, int] = {}
            for src in list(self.sid_alias.keys()):
                canon = self.canonical_sid(src)
                if canon == src:
                    continue
                out[int(src)] = int(canon)
            return out

    def set_alias(
        self,
        a: int,
        b: int,
        *,
        canonical: Optional[int] = None,
        append_embeddings: Optional[bool] = None,
        force: bool = False,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        now_ts = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            try:
                a_int = int(a)
                b_int = int(b)
            except Exception:
                return {
                    "src": int(a) if isinstance(a, int) else 0,
                    "dst": int(b) if isinstance(b, int) else 0,
                    "canonical": 0,
                    "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                    "applied": False,
                    "reason": "invalid_ids",
                }
            if a_int <= 0 or b_int <= 0:
                return {
                    "src": int(a_int),
                    "dst": int(b_int),
                    "canonical": 0,
                    "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                    "applied": False,
                    "reason": "invalid_ids",
                }
            root_a = self.canonical_sid(a_int)
            root_b = self.canonical_sid(b_int)
            if root_a == root_b:
                return {
                    "src": int(root_a),
                    "dst": int(root_b),
                    "canonical": int(root_a),
                    "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                    "applied": False,
                    "reason": "already_aliased",
                }
            if canonical is not None:
                try:
                    canonical_int = int(canonical)
                except Exception:
                    return {
                        "src": int(root_a),
                        "dst": int(root_b),
                        "canonical": int(root_a),
                        "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                        "applied": False,
                        "reason": "invalid_canonical",
                    }
                canon_root = self.canonical_sid(canonical_int)
                if canon_root not in (root_a, root_b):
                    return {
                        "src": int(root_a),
                        "dst": int(root_b),
                        "canonical": int(canon_root),
                        "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                        "applied": False,
                        "reason": "canonical_not_in_pair",
                    }
                dst_root = canon_root
            else:
                dst_root = min(root_a, root_b)
            src_root = root_b if dst_root == root_a else root_a

            if not force:
                if self.active_zones.get(root_a) and self.active_zones.get(root_b):
                    return {
                        "src": int(src_root),
                        "dst": int(dst_root),
                        "canonical": int(dst_root),
                        "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                        "applied": False,
                        "reason": "both_active",
                    }
                if self.was_copresent_recently(root_a, root_b, now_ts):
                    return {
                        "src": int(src_root),
                        "dst": int(dst_root),
                        "canonical": int(dst_root),
                        "append_embeddings": bool(append_embeddings) if append_embeddings is not None else self.alias_append_default,
                        "applied": False,
                        "reason": "copresent_recently",
                    }

            self.sid_alias[int(src_root)] = int(dst_root)
            if append_embeddings is None:
                append_embeddings = self.alias_append_default
            append_embeddings = bool(append_embeddings)

            if append_embeddings:
                src_entries = list(self.gallery.get(int(src_root), []))
                dst_entries = list(self.gallery.get(int(dst_root), []))
                if src_entries:
                    maxlen = self.gallery[int(dst_root)].maxlen or self.gallery_size
                    combined = dst_entries + src_entries
                    combined.sort(key=lambda item: float(item[0]), reverse=True)
                    self.gallery[int(dst_root)] = deque(combined[:maxlen], maxlen=maxlen)
                    embs = [emb for (_ts, emb) in self.gallery[int(dst_root)] if emb is not None]
                    if embs:
                        centroid = np.mean(np.stack(embs, axis=0), axis=0)
                        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                        self.sid_centroid[int(dst_root)] = centroid.astype(np.float32)
                    else:
                        self.sid_centroid.pop(int(dst_root), None)

                src_pose = list(self.pose_gallery.get(int(src_root), []))
                dst_pose = list(self.pose_gallery.get(int(dst_root), []))
                if src_pose:
                    maxlen = self.pose_gallery[int(dst_root)].maxlen or self.pose_gallery_size
                    combined_pose = dst_pose + src_pose
                    combined_pose.sort(key=lambda item: float(item[0]), reverse=True)
                    self.pose_gallery[int(dst_root)] = deque(combined_pose[:maxlen], maxlen=maxlen)
                    pose_vecs = [v for (_ts, v) in self.pose_gallery[int(dst_root)] if v is not None]
                    if pose_vecs:
                        centroid = np.mean(np.stack(pose_vecs, axis=0), axis=0)
                        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                        self.pose_centroid[int(dst_root)] = centroid.astype(np.float32)
                    else:
                        self.pose_centroid.pop(int(dst_root), None)
                if src_root in self.pose_last_seen or dst_root in self.pose_last_seen:
                    self.pose_last_seen[int(dst_root)] = max(
                        float(self.pose_last_seen.get(int(src_root), 0.0)),
                        float(self.pose_last_seen.get(int(dst_root), 0.0)),
                    )

            # Merge last-seen / appearance state by most-recent timestamp.
            last_src = self.sid_global_last_seen.get(int(src_root))
            last_dst = self.sid_global_last_seen.get(int(dst_root))
            if last_src is not None or last_dst is not None:
                if last_dst is None or (last_src is not None and float(last_src) > float(last_dst)):
                    if int(src_root) in self.sid_last_bbox:
                        self.sid_last_bbox[int(dst_root)] = self.sid_last_bbox[int(src_root)]
                    if int(src_root) in self.sid_last_brightness:
                        self.sid_last_brightness[int(dst_root)] = self.sid_last_brightness[int(src_root)]
                    if int(src_root) in self.sid_last_color:
                        try:
                            self.sid_last_color[int(dst_root)] = self.sid_last_color[int(src_root)].copy()
                        except Exception:
                            self.sid_last_color[int(dst_root)] = self.sid_last_color[int(src_root)]
                if last_src is None:
                    self.sid_global_last_seen[int(dst_root)] = float(last_dst)
                elif last_dst is None:
                    self.sid_global_last_seen[int(dst_root)] = float(last_src)
                else:
                    self.sid_global_last_seen[int(dst_root)] = float(max(last_src, last_dst))

            # Preserve earliest first-seen.
            first_src = self.sid_global_first_seen.get(int(src_root), float("inf"))
            first_dst = self.sid_global_first_seen.get(int(dst_root), float("inf"))
            earliest = min(float(first_src), float(first_dst))
            if math.isfinite(earliest):
                self.sid_global_first_seen[int(dst_root)] = float(earliest)

            # Canonicalize in-flight state (active tracks/zones).
            new_active_zones: Dict[int, set] = defaultdict(set)
            for (s_id, ds_id), rec in self.active_tracks.items():
                try:
                    old_sid = int(rec.get("stable_id"))
                except Exception:
                    old_sid = None
                if old_sid is None:
                    continue
                new_sid = self.canonical_sid(old_sid)
                if new_sid != old_sid:
                    rec["stable_id"] = int(new_sid)
                zone = rec.get("zone", "default")
                new_active_zones[int(new_sid)].add((int(s_id), zone))
            self.active_zones = new_active_zones

            # Canonicalize ghosts.
            for dq in self.ghosts.values():
                for ghost in dq:
                    try:
                        ghost["stable_id"] = self.canonical_sid(int(ghost.get("stable_id")))
                    except Exception:
                        pass

            self._refresh_alias_reserved()
            if self.sid_alias_reserved:
                self._free_sids = [sid for sid in self._free_sids if sid not in self.sid_alias_reserved]
                heapq.heapify(self._free_sids)
                self._free_sids_set = set(self._free_sids)

            self._record_alias_event(
                "merge",
                int(src_root),
                int(dst_root),
                ts=float(now_ts),
                extra={"triggered_by": "api"},
            )
            self._save_aliases()
            return {
                "src": int(src_root),
                "dst": int(dst_root),
                "canonical": int(dst_root),
                "append_embeddings": append_embeddings,
                "applied": True,
                "reason": None,
            }

    def unset_alias(self, src: int) -> Dict[str, Any]:
        with self._lock:
            try:
                src_int = int(src)
            except Exception:
                return {"src": int(src) if isinstance(src, int) else 0, "removed": False, "reason": "invalid_src"}
            removed = False
            if src_int in self.sid_alias:
                self.sid_alias.pop(src_int, None)
                removed = True
            self._refresh_alias_reserved()
            if self.sid_alias_reserved:
                self._free_sids = [sid for sid in self._free_sids if sid not in self.sid_alias_reserved]
                heapq.heapify(self._free_sids)
                self._free_sids_set = set(self._free_sids)
            self._record_alias_event("unset", int(src_int), None, ts=time.time(), extra={"triggered_by": "api"})
            self._save_aliases()
            return {
                "src": int(src_int),
                "removed": bool(removed),
                "reason": None if removed else "not_found",
            }

    def clear_aliases(self) -> int:
        with self._lock:
            count = len(self.sid_alias)
            self.sid_alias.clear()
            self._refresh_alias_reserved()
            self._record_alias_event("clear_all", 0, None, ts=time.time(), extra={"count": int(count)})
            self._save_aliases()
            return int(count)

    def set_aliases_batch(
        self,
        pairs: List[Dict[str, Any]],
        *,
        force: bool = False,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        now_ts = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            if not isinstance(pairs, list) or not pairs:
                return {"results": [], "applied_count": 0, "failed_count": 0}

            parsed: List[Dict[str, Any]] = []
            for idx, item in enumerate(pairs):
                if not isinstance(item, dict):
                    parsed.append({"index": idx, "invalid_reason": "invalid_pair"})
                    continue
                try:
                    a = int(item.get("a"))
                    b = int(item.get("b"))
                except Exception:
                    parsed.append({"index": idx, "invalid_reason": "invalid_pair"})
                    continue
                if a <= 0 or b <= 0:
                    parsed.append({"index": idx, "invalid_reason": "invalid_pair", "a": a, "b": b})
                    continue
                canonical = item.get("canonical")
                canonical_root = None
                if canonical is not None:
                    try:
                        canonical_root = self.canonical_sid(int(canonical))
                    except Exception:
                        canonical_root = None
                append_embeddings = item.get("append_embeddings")
                parsed.append(
                    {
                        "index": idx,
                        "a": a,
                        "b": b,
                        "canonical": canonical_root,
                        "append_embeddings": append_embeddings,
                        "invalid_reason": None,
                    }
                )

            # Build components on current canonical roots.
            adjacency: Dict[int, set] = defaultdict(set)
            for entry in parsed:
                if entry.get("invalid_reason"):
                    continue
                root_a = self.canonical_sid(entry["a"])
                root_b = self.canonical_sid(entry["b"])
                entry["root_a"] = root_a
                entry["root_b"] = root_b
                if root_a == root_b:
                    continue
                adjacency[root_a].add(root_b)
                adjacency[root_b].add(root_a)

            comp_id: Dict[int, int] = {}
            components: List[List[int]] = []
            for node in adjacency.keys():
                if node in comp_id:
                    continue
                stack = [node]
                comp = []
                comp_idx = len(components)
                while stack:
                    cur = stack.pop()
                    if cur in comp_id:
                        continue
                    comp_id[cur] = comp_idx
                    comp.append(cur)
                    for nxt in adjacency.get(cur, set()):
                        if nxt not in comp_id:
                            stack.append(nxt)
                components.append(comp)

            comp_canon: Dict[int, int] = {}
            conflicts: List[Dict[str, Any]] = []
            for idx, comp in enumerate(components):
                canonicals = set()
                for entry in parsed:
                    if entry.get("invalid_reason"):
                        continue
                    if entry.get("root_a") in comp or entry.get("root_b") in comp:
                        canon_root = entry.get("canonical")
                        if canon_root is not None:
                            if canon_root not in comp:
                                conflicts.append(
                                    {
                                        "index": entry.get("index"),
                                        "reason": "canonical_not_in_component",
                                        "canonical": canon_root,
                                    }
                                )
                            else:
                                canonicals.add(int(canon_root))
                if len(canonicals) > 1:
                    conflicts.append(
                        {
                            "component": idx,
                            "reason": "conflicting_canonical",
                            "canonicals": sorted(list(canonicals)),
                        }
                    )
                if canonicals:
                    comp_canon[idx] = int(sorted(list(canonicals))[0])
                else:
                    comp_canon[idx] = int(min(comp))

            if conflicts:
                results = []
                for entry in parsed:
                    if entry.get("invalid_reason"):
                        reason = entry.get("invalid_reason")
                    else:
                        reason = "conflicting_canonical"
                    results.append(
                        {
                            "src": int(entry.get("root_a", entry.get("a", 0)) or 0),
                            "dst": int(entry.get("root_b", entry.get("b", 0)) or 0),
                            "canonical": int(entry.get("canonical") or 0),
                            "append_embeddings": bool(
                                entry.get("append_embeddings")
                                if entry.get("append_embeddings") is not None
                                else self.alias_append_default
                            ),
                            "applied": False,
                            "reason": reason,
                        }
                    )
                return {
                    "results": results,
                    "applied_count": 0,
                    "failed_count": len(results),
                    "error": "conflicting_canonical",
                    "conflicts": conflicts,
                }

            # Cycle detection using proposed edges (current alias + batch proposals).
            proposed: Dict[int, int] = {}
            def _would_create_cycle(src: int, dst: int) -> bool:
                curr = dst
                visited: set[int] = set()
                for _ in range(64):
                    if curr == src:
                        return True
                    if curr in visited:
                        break
                    visited.add(curr)
                    nxt = proposed.get(curr)
                    if nxt is None:
                        nxt = self.sid_alias.get(curr)
                    if nxt is None:
                        break
                    curr = int(nxt)
                return False

            for entry in parsed:
                if entry.get("invalid_reason"):
                    continue
                root_a = entry.get("root_a")
                root_b = entry.get("root_b")
                if root_a is None or root_b is None or root_a == root_b:
                    continue
                canon = entry.get("canonical")
                if canon in (root_a, root_b):
                    dst = canon
                    src = root_b if dst == root_a else root_a
                else:
                    # Use input order for cycle validation (a -> b).
                    src = root_a
                    dst = root_b
                if _would_create_cycle(int(src), int(dst)):
                    entry["invalid_reason"] = "cycle_detected"
                    continue
                proposed[int(src)] = int(dst)

            results: List[Dict[str, Any]] = []
            autosave = self.alias_autosave
            self.alias_autosave = False
            try:
                for entry in parsed:
                    if entry.get("invalid_reason"):
                        results.append(
                            {
                                "src": int(entry.get("root_a", entry.get("a", 0)) or 0),
                                "dst": int(entry.get("root_b", entry.get("b", 0)) or 0),
                                "canonical": int(entry.get("canonical") or 0),
                                "append_embeddings": bool(
                                    entry.get("append_embeddings")
                                    if entry.get("append_embeddings") is not None
                                    else self.alias_append_default
                                ),
                                "applied": False,
                                "reason": entry.get("invalid_reason"),
                            }
                        )
                        continue
                    a = int(entry["a"])
                    b = int(entry["b"])
                    root_a = self.canonical_sid(a)
                    root_b = self.canonical_sid(b)
                    if root_a == root_b:
                        results.append(
                            {
                                "src": int(root_a),
                                "dst": int(root_b),
                                "canonical": int(root_a),
                                "append_embeddings": bool(
                                    entry.get("append_embeddings")
                                    if entry.get("append_embeddings") is not None
                                    else self.alias_append_default
                                ),
                                "applied": False,
                                "reason": "already_aliased",
                            }
                        )
                        continue
                    comp = comp_id.get(root_a)
                    canon = comp_canon.get(comp) if comp is not None else None
                    canonical_arg = None
                    if canon in (root_a, root_b):
                        canonical_arg = canon
                    elif entry.get("canonical") in (root_a, root_b):
                        canonical_arg = entry.get("canonical")
                    res = self.set_alias(
                        root_a,
                        root_b,
                        canonical=canonical_arg,
                        append_embeddings=entry.get("append_embeddings"),
                        force=force,
                        now_ts=now_ts,
                    )
                    results.append(res)
            finally:
                self.alias_autosave = autosave
            if autosave:
                self._save_aliases()
            applied_count = sum(1 for r in results if r.get("applied"))
            failed_count = max(0, len(results) - applied_count)
            return {
                "results": results,
                "applied_count": int(applied_count),
                "failed_count": int(failed_count),
            }

    def suggest_aliases(
        self,
        *,
        min_sim: Optional[float] = None,
        limit: int = 20,
        require_inactive: bool = True,
        now_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        now_ts = float(now_ts if now_ts is not None else time.time())
        min_sim = float(self.suggest_min_sim if min_sim is None else min_sim)
        with self._lock:
            # Build canonical SID set with enough support.
            candidates: List[int] = []
            seen: set[int] = set()
            for sid in list(self.gallery.keys()):
                sid_int = int(sid)
                sid_can = self.canonical_sid(sid_int)
                if sid_can in seen:
                    continue
                if self._is_alias_src(sid_int):
                    continue
                seen.add(sid_can)
                if len(self.gallery.get(int(sid_can), [])) < int(self.min_embeddings_for_suggest):
                    continue
                candidates.append(int(sid_can))

            centroids: Dict[int, np.ndarray] = {}
            for sid in candidates:
                centroid = self.sid_centroid.get(int(sid))
                if centroid is None:
                    vecs = [v for (_ts, v) in self.gallery.get(int(sid), []) if v is not None]
                    if not vecs:
                        continue
                    centroid = np.mean(np.stack(vecs, axis=0), axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                centroids[int(sid)] = centroid
            candidates = [sid for sid in candidates if sid in centroids]

            best: Dict[int, Tuple[int, float, float]] = {}
            for sid in candidates:
                sims: List[Tuple[float, int]] = []
                for other in candidates:
                    if other == sid:
                        continue
                    sim = self._cosine(centroids[sid], centroids[other])
                    sims.append((float(sim), int(other)))
                if not sims:
                    continue
                sims.sort(key=lambda item: item[0], reverse=True)
                best_sid = int(sims[0][1])
                best_sim = float(sims[0][0])
                second_sim = float(sims[1][0]) if len(sims) > 1 else -1.0
                best[int(sid)] = (best_sid, best_sim, second_sim)

            results: List[Dict[str, Any]] = []
            for sid, (best_sid, best_sim, second_sim) in best.items():
                if best_sim - float(second_sim) < float(self.suggest_mnn_margin):
                    continue
                if best.get(best_sid, (None, 0.0, 0.0))[0] != sid:
                    continue
                if sid > best_sid:
                    continue

                pose_sim = self._pose_similarity_between_sids(sid, best_sid)
                sim_ok = best_sim >= float(min_sim)
                if pose_sim is not None and best_sim < 0.94 and float(pose_sim) >= float(self.suggest_pose_sim_high):
                    sim_ok = True
                if not sim_ok:
                    continue

                blocked = False
                block_reason: Optional[str] = None
                if self.active_zones.get(int(sid)) and self.active_zones.get(int(best_sid)):
                    blocked = True
                    block_reason = "both_active"
                elif self.was_copresent_recently(int(sid), int(best_sid), float(now_ts)):
                    blocked = True
                    block_reason = "copresent_recently"
                elif require_inactive:
                    if self.active_zones.get(int(sid)) or self.active_zones.get(int(best_sid)):
                        blocked = True
                        block_reason = "require_inactive"

                if pose_sim is not None and best_sim >= 0.94 and float(pose_sim) < float(self.suggest_pose_sim_low):
                    blocked = True
                    block_reason = "pose_mismatch"

                count_a = len(self.gallery.get(int(sid), []))
                count_b = len(self.gallery.get(int(best_sid), []))
                if count_a > count_b:
                    preferred = int(sid)
                elif count_b > count_a:
                    preferred = int(best_sid)
                else:
                    first_a = float(self.sid_global_first_seen.get(int(sid), float("inf")))
                    first_b = float(self.sid_global_first_seen.get(int(best_sid), float("inf")))
                    if first_a < first_b:
                        preferred = int(sid)
                    elif first_b < first_a:
                        preferred = int(best_sid)
                    else:
                        preferred = int(min(sid, best_sid))

                results.append(
                    {
                        "a": int(sid),
                        "b": int(best_sid),
                        "sim": float(best_sim),
                        "pose_sim": float(pose_sim) if pose_sim is not None else None,
                        "canonical": int(min(sid, best_sid)),
                        "preferred_canonical": int(preferred),
                        "a_embedding_count": int(count_a),
                        "b_embedding_count": int(count_b),
                        "blocked": bool(blocked),
                        "block_reason": block_reason,
                    }
                )

            results.sort(key=lambda item: float(item.get("sim", 0.0)), reverse=True)
            if limit > 0:
                results = results[: int(limit)]
            return results

    def update(
        self,
        sensor_id: int,
        ds_obj_id: int,
        bbox_ltrbwh: BBox,
        ts: float,
        zone: Optional[str],
        frame_bgr: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        pose_features: Optional[Dict[str, float]] = None,
        pose_quality: Optional[Dict[str, float]] = None,
    ) -> int:
        """Update or create stable_id for a DS track.

        If new, attempt to match ghost for same camera, else gallery (cross-camera).
        Optionally extracts embedding when due (new track or embed interval elapsed).
        """
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            is_new = rec is None
            if rec is not None and self.aliases_enabled:
                try:
                    old_sid = int(rec.get("stable_id"))
                    new_sid = self.canonical_sid(old_sid)
                except Exception:
                    old_sid = None
                    new_sid = None
                if old_sid is not None and new_sid is not None and new_sid != old_sid:
                    zone_existing = rec.get("zone", "default")
                    try:
                        pairs = self.active_zones.get(old_sid, set())
                        if (int(sensor_id), zone_existing) in pairs:
                            pairs.remove((int(sensor_id), zone_existing))
                        if pairs:
                            self.active_zones[old_sid] = pairs
                        else:
                            self.active_zones.pop(old_sid, None)
                    except Exception:
                        pass
                    rec["stable_id"] = int(new_sid)
                    self.active_zones[int(new_sid)].add((int(sensor_id), zone_existing))

            emb: Optional[np.ndarray] = None
            curr_brightness: Optional[float] = None
            curr_color: Optional[np.ndarray] = None
            need_embed = False
            if is_new:
                need_embed = True
            else:
                last_emb_ts = rec.get("last_emb_ts", 0.0)
                if ts - float(last_emb_ts) >= self.embed_interval_s:
                    need_embed = True

            # Prefer caller-supplied embeddings (e.g., from DeepStream SGIE tensor meta).
            if need_embed and embedding is not None:
                try:
                    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
                    if vec.size > 0:
                        n = float(np.linalg.norm(vec) + 1e-12)
                        emb = (vec / n).astype(np.float32)
                except Exception:
                    emb = None
            elif need_embed and self._use_extractor:
                crop = self._crop(frame_bgr, bbox_ltrbwh, expand=self.crop_expand)
                if crop is not None:
                    # Quality gating to avoid embedding drift on tiny/blurred crops
                    if crop.shape[0] < self.min_crop_h:
                        emb = None
                    else:
                        if self._crop_blur_var(crop) < self.min_laplacian_var:
                            emb = None
                        else:
                            emb = self._extract_embedding_from_crop(crop)
                    # Record brightness for adaptive penalties
                    if emb is not None:
                        curr_brightness = self._crop_brightness(crop)
                    # Compute color hist for additional discrimination
                    curr_color = self._crop_color_hist(crop)

            pose_vec: Optional[np.ndarray] = None
            pose_valid = False
            if self.pose_enabled:
                need_pose = False
                if is_new:
                    need_pose = True
                else:
                    last_pose_ts = rec.get("last_pose_ts", 0.0)
                    if rec.get("pose_vec") is None:
                        need_pose = True
                    else:
                        interval = float(self.pose_interval_s)
                        if interval > 0.0 and (ts - float(last_pose_ts)) >= interval:
                            need_pose = True
                if need_pose and pose_features is not None:
                    try:
                        vec = self._pose_vector_from_features(pose_features)
                    except Exception:
                        vec = None
                    if vec is not None and self._pose_quality_ok(pose_quality):
                        pose_vec = vec
                        pose_valid = True

            # New track: try to match
            if is_new:
                sid = None
                # Prefer ghost match (same camera, recent disappearance)
                if emb is not None or pose_valid:
                    sid = self._match_ghost(
                        sensor_id,
                        emb,
                        bbox_ltrbwh,
                        ts,
                        pose_vec=pose_vec if pose_valid else None,
                    )
                    # Cross-camera active/gallery match if enabled
                    if sid is None and emb is not None:
                        g_id, g_reid, g_req = self._gallery_best(
                            emb,
                            sensor_id=int(sensor_id),
                            curr_bbox=bbox_ltrbwh,
                            curr_brightness=curr_brightness,
                            curr_color=curr_color,
                            pose_vec=pose_vec if pose_valid else None,
                            pose_valid=pose_valid,
                            now_ts=float(ts),
                        )
                        if g_id is not None and float(g_reid) >= float(g_req):
                            # If the candidate stable_id is already active on this sensor,
                            # optionally require a small extra margin to avoid merging co-present people.
                            can_take = True
                            if self.active_id_guard_strict:
                                active_pairs = self.active_zones.get(int(g_id), set())
                                active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
                                if active_here and (float(g_reid) < (float(g_req) + self.active_id_guard_margin)):
                                    can_take = False
                            if can_take and (self.allow_multi_zone_active or not self.active_zones.get(g_id)):
                                sid = g_id
                    if sid is None and emb is None and pose_valid and pose_vec is not None:
                        g_id, g_pose, g_req = self._pose_gallery_best(pose_vec, now_ts=float(ts))
                        if g_id is not None and self.aliases_enabled:
                            g_id = self.canonical_sid(int(g_id))
                        if g_id is not None and float(g_pose) >= float(g_req):
                            can_take = True
                            if self.active_id_guard_strict:
                                active_pairs = self.active_zones.get(int(g_id), set())
                                active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
                                if active_here and (float(g_pose) < (float(g_req) + self.active_id_guard_margin)):
                                    can_take = False
                            if can_take and (self.allow_multi_zone_active or not self.active_zones.get(g_id)):
                                sid = g_id

                if sid is None:
                    # Global new-ID hysteresis + soft-cap handling
                    active_ids_here = {s for s, pairs in self.active_zones.items() if any(int(x) == int(sensor_id) for (x, _z) in pairs)}
                    at_cap = len(active_ids_here) >= self.max_active_ids_per_sensor
                    required = max(1, self.new_id_hysteresis_frames)
                    if at_cap:
                        required = max(required, self.new_id_confirm_frames_at_cap)
                    cnt = self._pending_new_counts.get(key, 0) + 1
                    self._pending_new_counts[key] = cnt
                    self._pending_new_ts[key] = float(ts)
                    if cnt < required:
                        # Defer *finalizing* the ID, but still return a positive stable_id so
                        # user-facing telemetry never needs to fall back to raw tracker IDs.
                        pending_sid = self._pending_new_sids.get(key)
                        if pending_sid is None:
                            pending_sid = int(self._alloc_sid())
                            self._pending_new_sids[key] = int(pending_sid)
                        if self.aliases_enabled:
                            return int(self.canonical_sid(int(pending_sid)))
                        return int(pending_sid)
                    # Confirmation reached: use the pending stable_id if one was allocated.
                    pending_sid = self._pending_new_sids.get(key)
                    if pending_sid is not None:
                        sid = int(pending_sid)
                    if at_cap and self.new_id_confirm_frames_at_cap > 0:
                        # Evict stale locals to free a slot
                        now = float(ts)
                        oldest_sid = None
                        oldest_age = -1.0
                        for sid_cand in list(active_ids_here):
                            ages = [now - rec2.get("last_seen_ts", now) for (s_id2, _ds2), rec2 in self.active_tracks.items() if int(s_id2) == int(sensor_id) and int(rec2.get("stable_id", -1)) == int(sid_cand)]
                            age = max(ages) if ages else 0.0
                            if age >= self.active_evict_grace_s and age > oldest_age:
                                oldest_age = age
                                oldest_sid = sid_cand
                        if oldest_sid is not None:
                            try:
                                pairs = self.active_zones.get(int(oldest_sid), set())
                                pairs = {p for p in pairs if int(p[0]) != int(sensor_id)}
                                if pairs:
                                    self.active_zones[int(oldest_sid)] = pairs
                                else:
                                    self.active_zones.pop(int(oldest_sid), None)
                            except Exception:
                                pass
                        # fall through to create new ID
                    # Try global reuse before creating when total exceeds soft-cap
                    total_ids = len(self.gallery)
                    if self.total_id_reuse and total_ids >= self.max_total_ids:
                        # Attempt to map into an existing identity with a slightly relaxed threshold
                        if emb is not None:
                            req2 = max(0.0, self.cos_sim_high_threshold - 0.04)
                            g_id2, g_reid2, _g_req2 = self._gallery_best(
                                emb,
                                sensor_id=int(sensor_id),
                                curr_bbox=bbox_ltrbwh,
                                curr_brightness=curr_brightness,
                                curr_color=curr_color,
                                pose_vec=pose_vec if pose_valid else None,
                                pose_valid=pose_valid,
                                now_ts=float(ts),
                                min_reid=req2,
                            )
                            if g_id2 is not None and float(g_reid2) >= float(req2):
                                sid = int(g_id2)
                        if sid is None:
                            # Recycle the least recently seen, fully inactive ID if old enough
                            candidates = [
                                int(s)
                                for s in self.gallery.keys()
                                if not self.active_zones.get(int(s)) and not self._is_alias_reserved(int(s))
                            ]
                            oldest_sid = None
                            oldest_age = -1.0
                            now = float(ts)
                            for s in candidates:
                                age = now - float(self.sid_global_last_seen.get(int(s), 0.0))
                                if age >= self.total_id_reuse_min_age_s and age > oldest_age:
                                    oldest_age = age
                                    oldest_sid = int(s)
                            if oldest_sid is not None:
                                # Clear per-id appearance except numeric id
                                try:
                                    self._purge_sid_state(int(oldest_sid))
                                except Exception:
                                    pass
                                sid = int(oldest_sid)
                    # Create new stable ID now if still none
                    if sid is None:
                        sid = self._alloc_sid()
                    # Clear pending counter after minting
                    try:
                        self._pending_new_counts.pop(key, None)
                        self._pending_new_ts.pop(key, None)
                        self._pending_new_sids.pop(key, None)
                    except Exception:
                        pass

                # Creating an active record: clear any pending state for this track key.
                try:
                    self._pending_new_counts.pop(key, None)
                    self._pending_new_ts.pop(key, None)
                    self._pending_new_sids.pop(key, None)
                except Exception:
                    pass

                if self.aliases_enabled:
                    sid = self.canonical_sid(int(sid))

                rec = {
                    "stable_id": int(sid),
                    "bbox": bbox_ltrbwh,
                    "last_seen_ts": float(ts),
                    "last_emb_ts": float(ts) if emb is not None else 0.0,
                    "emb": emb,
                    "pose_vec": pose_vec,
                    "last_pose_ts": float(ts) if pose_vec is not None else 0.0,
                    "zone": zone or "default",
                }
                self.active_tracks[key] = rec
                # Track zones and gallery
                self.active_zones[int(sid)].add((int(sensor_id), rec["zone"]))
                if emb is not None:
                    sid_int = int(sid)
                    self.gallery[sid_int].append((float(ts), emb))
                    if sid_int not in self.sid_global_first_seen:
                        self.sid_global_first_seen[sid_int] = float(ts)
                    # Initialize EMA centroid with first embedding
                    try:
                        newc = emb / (np.linalg.norm(emb) + 1e-12)
                        self.sid_centroid[sid_int] = newc.astype(np.float32)
                    except Exception:
                        pass
                    self.sid_last_bbox[sid_int] = bbox_ltrbwh
                    if curr_brightness is not None:
                        self.sid_last_brightness[sid_int] = curr_brightness
                    if curr_color is not None:
                        self.sid_last_color[sid_int] = curr_color.astype(np.float32)
                if pose_vec is not None:
                    self._update_pose_state(int(sid), pose_vec, float(ts))
                return int(sid)

            # Existing track: update
            rec["bbox"] = bbox_ltrbwh
            rec["last_seen_ts"] = float(ts)
            if zone:
                rec["zone"] = zone
                self.active_zones[int(rec["stable_id"])].add((int(sensor_id), rec["zone"]))
            if emb is not None:
                # If we minted an allocator-only ID before an embedding arrived (external-embedding mode),
                # try to reconcile the first embedding against ghosts/gallery and optionally remap to the
                # matched stable_id. This keeps cross-camera matching viable even when the first frame(s)
                # lacked tensor meta.
                if rec.get("emb") is None:
                    try:
                        current_sid = int(rec.get("stable_id"))
                    except Exception:
                        current_sid = int(rec["stable_id"])
                    candidate_sid = None
                    try:
                        candidate_sid = self._match_ghost(
                            sensor_id,
                            emb,
                            bbox_ltrbwh,
                            ts,
                            pose_vec=pose_vec if pose_valid else None,
                        )
                    except Exception:
                        candidate_sid = None
                    if candidate_sid is None:
                        try:
                            g_id, g_reid, g_req = self._gallery_best(
                                emb,
                                sensor_id=int(sensor_id),
                                curr_bbox=bbox_ltrbwh,
                                curr_brightness=curr_brightness,
                                curr_color=curr_color,
                                pose_vec=pose_vec if pose_valid else None,
                                pose_valid=pose_valid,
                                now_ts=float(ts),
                            )
                        except Exception:
                            g_id, g_reid, g_req = None, -1.0, self.cos_sim_high_threshold
                        if g_id is not None and float(g_reid) >= float(g_req):
                            can_take = True
                            if self.active_id_guard_strict:
                                active_pairs = self.active_zones.get(int(g_id), set())
                                active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
                                if active_here and (float(g_reid) < (float(g_req) + self.active_id_guard_margin)):
                                    can_take = False
                            if can_take and (self.allow_multi_zone_active or not self.active_zones.get(g_id)):
                                candidate_sid = int(g_id)
                    if candidate_sid is not None and self.aliases_enabled:
                        candidate_sid = self.canonical_sid(int(candidate_sid))
                    if candidate_sid is not None and int(candidate_sid) > 0 and int(candidate_sid) != int(current_sid):
                        old_sid = int(current_sid)
                        new_sid = int(candidate_sid)
                        try:
                            old_zone = rec.get("zone", "default")
                            pairs = self.active_zones.get(old_sid, set())
                            if (int(sensor_id), old_zone) in pairs:
                                pairs.remove((int(sensor_id), old_zone))
                            if pairs:
                                self.active_zones[old_sid] = pairs
                            else:
                                self.active_zones.pop(old_sid, None)
                        except Exception:
                            pass
                        rec["stable_id"] = int(new_sid)
                        self.active_zones[int(new_sid)].add((int(sensor_id), rec.get("zone", "default")))
                        # If the old ID had no appearance state and is now unused, recycle it immediately.
                        try:
                            still_used = any(int(r.get("stable_id", -1)) == old_sid for r in self.active_tracks.values())
                        except Exception:
                            still_used = True
                        if (
                            not still_used
                            and (old_sid not in self.gallery)
                            and (old_sid not in self.pose_gallery)
                            and not self._is_alias_reserved(int(old_sid))
                        ):
                            self._purge_sid_state(old_sid)
                            self._free_sid(old_sid)

                rec["emb"] = emb
                rec["last_emb_ts"] = float(ts)
                sid_int = int(rec["stable_id"]) 
                self.gallery[sid_int].append((float(ts), emb))
                if sid_int not in self.sid_global_first_seen:
                    self.sid_global_first_seen[sid_int] = float(ts)
                # EMA centroid update
                try:
                    old = self.sid_centroid.get(sid_int)
                    if old is None:
                        newc = emb
                    else:
                        newc = (1.0 - self.ema_alpha) * old + self.ema_alpha * emb
                    newc = newc / (np.linalg.norm(newc) + 1e-12)
                    self.sid_centroid[sid_int] = newc.astype(np.float32)
                except Exception:
                    pass
                self.sid_last_bbox[sid_int] = bbox_ltrbwh
                if curr_brightness is not None:
                    self.sid_last_brightness[sid_int] = curr_brightness
                if curr_color is not None:
                    self.sid_last_color[sid_int] = curr_color.astype(np.float32)
            if pose_vec is not None and pose_valid:
                if emb is None and rec.get("pose_vec") is None:
                    try:
                        current_sid = int(rec.get("stable_id"))
                    except Exception:
                        current_sid = int(rec["stable_id"])
                    candidate_sid = None
                    try:
                        candidate_sid = self._match_ghost(
                            sensor_id,
                            None,
                            bbox_ltrbwh,
                            ts,
                            pose_vec=pose_vec,
                        )
                    except Exception:
                        candidate_sid = None
                    if candidate_sid is None:
                        try:
                            g_id, g_pose, g_req = self._pose_gallery_best(pose_vec, now_ts=float(ts))
                        except Exception:
                            g_id, g_pose, g_req = None, -1.0, self.pose_only_threshold
                        if g_id is not None and self.aliases_enabled:
                            g_id = self.canonical_sid(int(g_id))
                        if g_id is not None and float(g_pose) >= float(g_req):
                            can_take = True
                            if self.active_id_guard_strict:
                                active_pairs = self.active_zones.get(int(g_id), set())
                                active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
                                if active_here and (float(g_pose) < (float(g_req) + self.active_id_guard_margin)):
                                    can_take = False
                            if can_take and (self.allow_multi_zone_active or not self.active_zones.get(g_id)):
                                candidate_sid = int(g_id)
                    if candidate_sid is not None and self.aliases_enabled:
                        candidate_sid = self.canonical_sid(int(candidate_sid))
                    if candidate_sid is not None and int(candidate_sid) > 0 and int(candidate_sid) != int(current_sid):
                        old_sid = int(current_sid)
                        new_sid = int(candidate_sid)
                        try:
                            old_zone = rec.get("zone", "default")
                            pairs = self.active_zones.get(old_sid, set())
                            if (int(sensor_id), old_zone) in pairs:
                                pairs.remove((int(sensor_id), old_zone))
                            if pairs:
                                self.active_zones[old_sid] = pairs
                            else:
                                self.active_zones.pop(old_sid, None)
                        except Exception:
                            pass
                        rec["stable_id"] = int(new_sid)
                        self.active_zones[int(new_sid)].add((int(sensor_id), rec.get("zone", "default")))
                        try:
                            still_used = any(int(r.get("stable_id", -1)) == old_sid for r in self.active_tracks.values())
                        except Exception:
                            still_used = True
                        if (
                            not still_used
                            and (old_sid not in self.gallery)
                            and (old_sid not in self.pose_gallery)
                            and not self._is_alias_reserved(int(old_sid))
                        ):
                            self._purge_sid_state(old_sid)
                            self._free_sid(old_sid)
                rec["pose_vec"] = pose_vec
                rec["last_pose_ts"] = float(ts)
                sid_int = int(rec["stable_id"])
                self._update_pose_state(sid_int, pose_vec, float(ts))
            self.active_tracks[key] = rec
            try:
                self.sid_global_last_seen[int(rec["stable_id"])] = float(ts)
            except Exception:
                pass
            if self.aliases_enabled:
                return int(self.canonical_sid(int(rec["stable_id"])))
            return int(rec["stable_id"])

    def needs_embedding(self, sensor_id: int, ds_obj_id: int, ts: float) -> bool:
        """Return True when an embedding update is due for the given track.

        - New tracks always need an embedding.
        - Tracks without an embedding need one.
        - In extractor mode, respect embed_interval_s exactly.
        - In external-embedding mode (use_extractor=False), treat embed_interval_s <= 0
          as "only once" (avoid re-consuming SGIE tensor outputs every frame).
        """
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            if rec is None:
                return True
            if rec.get("emb") is None:
                return True
            last_emb_ts = float(rec.get("last_emb_ts", 0.0) or 0.0)
            interval = float(self.embed_interval_s)
            if not self._use_extractor and interval <= 0.0:
                return False
            return (float(ts) - last_emb_ts) >= interval

    def needs_pose_update(self, sensor_id: int, ds_obj_id: int, ts: float) -> bool:
        """Return True when a pose-feature update is due for the given track."""
        if not self.pose_enabled:
            return False
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            if rec is None:
                return True
            if rec.get("pose_vec") is None:
                return True
            last_pose_ts = float(rec.get("last_pose_ts", 0.0) or 0.0)
            interval = float(self.pose_interval_s)
            if interval <= 0.0:
                return False
            return (float(ts) - last_pose_ts) >= interval

    def remove_missing_tracks(self, sensor_id: int, present_ds_ids: List[int], ts: float) -> None:
        """Move tracks not present this frame to ghost lists and update active_zones.

        Called at end of a frame for a specific sensor.
        """
        with self._lock:
            present_set = set(int(i) for i in present_ds_ids)
            # Also prune any pending new-track state for IDs that vanished before confirmation.
            try:
                pending_keys = [
                    k
                    for k in list(self._pending_new_ts.keys())
                    if int(k[0]) == int(sensor_id) and int(k[1]) not in present_set
                ]
                for k in pending_keys:
                    self._pending_new_counts.pop(k, None)
                    self._pending_new_ts.pop(k, None)
                    self._pending_new_sids.pop(k, None)
            except Exception:
                pass
            to_remove: List[Tuple[int, int]] = []
            for (s_id, ds_id), rec in self.active_tracks.items():
                if int(s_id) != int(sensor_id):
                    continue
                if int(ds_id) not in present_set:
                    to_remove.append((s_id, ds_id))

            for key in to_remove:
                rec = self.active_tracks.pop(key, None)
                if rec is None:
                    continue
                sid = int(rec["stable_id"])
                zone = rec.get("zone", "default")
                # Update active zones
                if (int(sensor_id), zone) in self.active_zones.get(sid, set()):
                    try:
                        self.active_zones[sid].remove((int(sensor_id), zone))
                    except Exception:
                        pass
                    if not self.active_zones[sid]:
                        self.active_zones.pop(sid, None)

                # Push to ghost list for this camera if we have appearance cues
                emb_val = rec.get("emb", None)
                pose_val = rec.get("pose_vec", None)
                if emb_val is not None or pose_val is not None:
                    ghost_sid = self.canonical_sid(sid) if self.aliases_enabled else sid
                    ghost_rec = {
                        "stable_id": ghost_sid,
                        "bbox": rec.get("bbox"),
                        "ts": float(ts),
                    }
                    if emb_val is not None:
                        ghost_rec["emb"] = emb_val
                    if pose_val is not None:
                        ghost_rec["pose"] = pose_val
                        ghost_rec["pose_ts"] = rec.get("last_pose_ts", float(ts))
                    self.ghosts[int(sensor_id)].append(ghost_rec)

    def prune_ghosts(self, now_ts: Optional[float] = None) -> None:
        with self._lock:
            t = float(now_ts if now_ts is not None else time.time())
            for sensor_id, dq in self.ghosts.items():
                while dq and (t - float(dq[0].get("ts", 0.0))) > self.max_ghost_age_s:
                    dq.popleft()
            # Return fully inactive SIDs to free-list after a cooldown (smarter reuse)
            try:
                active_sids = {int(rec.get("stable_id")) for rec in self.active_tracks.values()}
                ghost_sids = set()
                for dq in self.ghosts.values():
                    for g in dq:
                        try:
                            ghost_sids.add(int(g.get("stable_id")))
                        except Exception:
                            pass
                for sid, last_seen in list(self.sid_global_last_seen.items()):
                    if sid in active_sids or sid in ghost_sids:
                        continue
                    if self._is_alias_reserved(int(sid)):
                        continue
                    if (t - float(last_seen)) >= max(2.0, self.active_evict_grace_s):
                        self._purge_sid_state(sid)
                        self._free_sid(sid)
            except Exception:
                pass
            # Prune pending new-track counters that haven't been seen recently.
            try:
                cutoff = t - max(2.0, float(self.active_evict_grace_s))
                for key, last_ts in list(self._pending_new_ts.items()):
                    if float(last_ts) < cutoff:
                        self._pending_new_ts.pop(key, None)
                        self._pending_new_counts.pop(key, None)
            except Exception:
                pass
            try:
                self._prune_pose_gallery(t)
            except Exception:
                pass

    # --------------- Telemetry ----------------
    def get_sid_metrics(self) -> Dict[str, Any]:
        """Return SID allocator and activity metrics for telemetry."""
        with self._lock:
            try:
                # Active unique SIDs across all sensors
                active_sids = set()
                active_by_sensor: Dict[int, set] = defaultdict(set)
                for (s_id, _ds), rec in self.active_tracks.items():
                    try:
                        sid = int(rec.get("stable_id"))
                        active_sids.add(sid)
                        active_by_sensor[int(s_id)].add(sid)
                    except Exception:
                        pass
                # Ghost unique SIDs
                ghost_sids = set()
                total_ghosts = 0
                for dq in self.ghosts.values():
                    total_ghosts += len(dq)
                    for g in dq:
                        try:
                            ghost_sids.add(int(g.get("stable_id")))
                        except Exception:
                            pass
                # Build per-sensor counts
                active_counts = {int(k): len(v) for k, v in active_by_sensor.items()}
                # Free pool size
                free_pool_size = len(getattr(self, "_free_sids_set", set()))
                pending_new = len(self._pending_new_counts)
                latency_vals = list(self._match_latency_ms)
                p50 = self._percentile(latency_vals, 50.0)
                p95 = self._percentile(latency_vals, 95.0)
                return {
                    "active_unique": len(active_sids),
                    "active_by_sensor": active_counts,
                    "ghost_unique": len(ghost_sids),
                    "ghost_entries": total_ghosts,
                    "gallery_ids": len(self.gallery),
                    "stableid_gallery_size": len(self.gallery),
                    "stableid_backend_mode": str(self._backend_mode),
                    "stableid_gpu_match_p50_ms": p50,
                    "stableid_gpu_match_p95_ms": p95,
                    "stableid_backend_last_error": self._backend_last_error,
                    "free_sid_pool_size": free_pool_size,
                    "next_sid": int(self.next_stable_id),
                    "pending_new_count": int(pending_new),
                }
            except Exception:
                return {}

    # --------------- Internal ---------------
    def _match_ghost(
        self,
        sensor_id: int,
        emb: Optional[np.ndarray],
        bbox: BBox,
        ts: float,
        *,
        pose_vec: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        dq = self.ghosts.get(int(sensor_id))
        if not dq:
            return None
        # Loose spatial check threshold based on bbox diagonal
        x, y, w, h = bbox
        diag = float((w ** 2 + h ** 2) ** 0.5)
        best_sid, best_score = None, -1.0
        for ghost in reversed(dq):  # newest first
            if float(ts) - float(ghost.get("ts", 0.0)) > self.max_ghost_age_s:
                continue
            g_sid = int(ghost.get("stable_id"))
            if self.aliases_enabled:
                g_sid = self.canonical_sid(g_sid)
            # Exclusivity only if multi-active not allowed
            if not self.allow_multi_zone_active and self.active_zones.get(g_sid):
                continue
            gx, gy, gw, gh = ghost.get("bbox", (0, 0, 0, 0))
            dist = float(((x - gx) ** 2 + (y - gy) ** 2) ** 0.5)
            if dist > 1.5 * diag:
                continue
            age = float(ts) - float(ghost.get("ts", 0.0))

            if emb is not None:
                g_emb = ghost.get("emb")
                if g_emb is None:
                    continue
                sim = self._cosine(emb, g_emb)
                thr = self.cos_sim_threshold + (self.ghost_extra_margin if age >= self.ghost_strict_age_s else 0.0)
                if sim < thr:
                    continue
                score = float(sim)
                if pose_vec is not None:
                    g_pose = ghost.get("pose")
                    g_pose_ts = ghost.get("pose_ts", ghost.get("ts", 0.0))
                    if g_pose is not None:
                        if self.pose_max_age_s <= 0.0 or (float(ts) - float(g_pose_ts)) <= float(self.pose_max_age_s):
                            pose_sim = self._cosine(pose_vec, g_pose)
                            if pose_sim >= float(self.pose_sim_threshold):
                                score = float(sim) + float(self.pose_weight) * float(pose_sim)
                if score > best_score:
                    best_sid, best_score = g_sid, score
                continue

            if pose_vec is not None:
                g_pose = ghost.get("pose")
                g_pose_ts = ghost.get("pose_ts", ghost.get("ts", 0.0))
                if g_pose is None:
                    continue
                if self.pose_max_age_s > 0.0 and (float(ts) - float(g_pose_ts)) > float(self.pose_max_age_s):
                    continue
                pose_sim = self._cosine(pose_vec, g_pose)
                req = float(self.pose_only_threshold)
                if age >= self.ghost_strict_age_s:
                    req = float(req + self.ghost_extra_margin)
                if pose_sim < req:
                    continue
                score = float(pose_sim)
                if score > best_score:
                    best_sid, best_score = g_sid, score
        return best_sid
