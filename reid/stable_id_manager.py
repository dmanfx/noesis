import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple, Any
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
        cos_sim_threshold: float = 0.72,
        cos_sim_high_threshold: float = 0.80,
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
        ghost_strict_age_s: float = 2.0,
        ghost_extra_margin: float = 0.03,
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
    ) -> None:
        self._lock = threading.RLock()
        self.extractor = EmbeddingExtractor(
            model_path=model_path,
            device=device,
            image_size=image_size,
            model_name=model_name,
        )

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

        # Active tracks: (sensor_id, ds_obj_id) -> record
        self.active_tracks: Dict[Tuple[int, int], Dict] = {}

        # Ghosts by camera: sensor_id -> deque of ghost records
        self.ghosts: Dict[int, Deque[Dict]] = defaultdict(lambda: deque(maxlen=ghost_queue_max))

        # Identity gallery: stable_id -> deque of recent embeddings
        self.gallery: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=gallery_size))
        # EMA centroid per stable id
        self.sid_centroid: Dict[int, np.ndarray] = {}

        # Where each identity is currently active: stable_id -> set of (sensor_id, zone)
        self.active_zones: Dict[int, set] = defaultdict(set)

        # Last known bbox/appearance per stable_id (for adaptive penalties)
        self.sid_last_bbox: Dict[int, BBox] = {}
        self.sid_last_brightness: Dict[int, float] = {}
        self.sid_last_color: Dict[int, np.ndarray] = {}
        # Global last seen timestamp per stable_id (any sensor)
        self.sid_global_last_seen: Dict[int, float] = {}
        # Pending new-ID confirmation counters at cap
        self._pending_new_counts: Dict[Tuple[int, int], int] = {}

        self.next_stable_id = 1
        # Free-list allocator state
        self._free_sids: List[int] = []
        self._free_sids_set: set[int] = set()
        self._load_sid_pool()

    # --------------- Allocator -----------------
    def _alloc_sid(self) -> int:
        if self._free_sids:
            sid = heapq.heappop(self._free_sids)
            try:
                self._free_sids_set.remove(sid)
            except KeyError:
                pass
            return int(sid)
        sid = self.next_stable_id
        self.next_stable_id += 1
        return int(sid)

    def _free_sid(self, sid: int) -> None:
        try:
            sid = int(sid)
        except Exception:
            return
        if sid <= 0:
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
                json.dump({'free_sids': pool}, f)
        except Exception:
            pass

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

    def _gallery_best(self, emb: np.ndarray, sensor_id: Optional[int] = None, curr_bbox: Optional[BBox] = None, curr_brightness: Optional[float] = None, curr_color: Optional[np.ndarray] = None) -> Tuple[Optional[int], float]:
        best_id, best_score = None, -1.0
        for sid, vecs in self.gallery.items():
            if not vecs:
                continue
            # Use EMA centroid when available; fallback to mean
            centroid = self.sid_centroid.get(int(sid))
            if centroid is None:
                centroid = np.mean(np.stack(vecs, axis=0), axis=0)
                norm = np.linalg.norm(centroid) + 1e-12
                centroid = centroid / norm
            sim = self._cosine(emb, centroid)

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

            if score > best_score:
                best_id, best_score = sid, score
        return best_id, best_score

    # --------------- Public API --------------
    def update(
        self,
        sensor_id: int,
        ds_obj_id: int,
        bbox_ltrbwh: BBox,
        ts: float,
        zone: Optional[str],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> int:
        """Update or create stable_id for a DS track.

        If new, attempt to match ghost for same camera, else gallery (cross-camera).
        Optionally extracts embedding when due (new track or embed interval elapsed).
        """
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            is_new = rec is None

            emb: Optional[np.ndarray] = None
            need_embed = False
            if is_new:
                need_embed = True
            else:
                last_emb_ts = rec.get("last_emb_ts", 0.0)
                if ts - float(last_emb_ts) >= self.embed_interval_s:
                    need_embed = True

            if need_embed:
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
                    else:
                        curr_brightness = None
                    # Compute color hist for additional discrimination
                    curr_color = self._crop_color_hist(crop)
                else:
                    curr_color = None
            else:
                curr_color = None

            # New track: try to match
            if is_new:
                sid = None
                # Prefer ghost match (same camera, recent disappearance)
                if emb is not None:
                    sid = self._match_ghost(sensor_id, emb, bbox_ltrbwh, ts)
                    # Cross-camera active/gallery match if enabled
                    if sid is None:
                        g_brightness = curr_brightness if curr_brightness is not None else None
                        g_id, g_sim = self._gallery_best(emb, sensor_id=int(sensor_id), curr_bbox=bbox_ltrbwh, curr_brightness=g_brightness, curr_color=curr_color)
                        if g_id is not None:
                            # Cross-camera handoff: lower threshold if same ID seen recently on another sensor
                            req = self.cos_sim_high_threshold
                            last_glob = self.sid_global_last_seen.get(int(g_id))
                            if last_glob is not None and (ts - float(last_glob)) <= self.xcam_handoff_window_s:
                                req = max(0.0, req - self.xcam_handoff_margin)
                            if g_sim >= req:
                                # If the candidate stable_id is already active on this sensor,
                                # optionally require a small extra margin to avoid merging co-present people.
                                can_take = True
                                if self.active_id_guard_strict:
                                    active_pairs = self.active_zones.get(int(g_id), set())
                                    active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
                                    if active_here and (g_sim < (req + self.active_id_guard_margin)):
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
                    if cnt < required:
                        # Defer ID creation; provisional negative label for UI stability
                        return int(-abs(int(ds_obj_id)))
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
                            g_id2, g_sim2 = self._gallery_best(emb, sensor_id=int(sensor_id), curr_bbox=bbox_ltrbwh, curr_brightness=curr_brightness, curr_color=curr_color)
                            req2 = max(0.0, self.cos_sim_high_threshold - 0.04)
                            if g_id2 is not None and g_sim2 >= req2:
                                sid = int(g_id2)
                        if sid is None:
                            # Recycle the least recently seen, fully inactive ID if old enough
                            candidates = [int(s) for s in self.gallery.keys() if not self.active_zones.get(int(s))]
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
                                    self.gallery.pop(int(oldest_sid), None)
                                    self.sid_centroid.pop(int(oldest_sid), None)
                                    self.sid_last_bbox.pop(int(oldest_sid), None)
                                    self.sid_last_brightness.pop(int(oldest_sid), None)
                                    self.sid_last_color.pop(int(oldest_sid), None)
                                except Exception:
                                    pass
                                sid = int(oldest_sid)
                    # Create new stable ID now if still none
                    if sid is None:
                        sid = self._alloc_sid()
                    # Clear pending counter after minting
                    try:
                        self._pending_new_counts.pop(key, None)
                    except Exception:
                        pass

                rec = {
                    "stable_id": int(sid),
                    "bbox": bbox_ltrbwh,
                    "last_seen_ts": float(ts),
                    "last_emb_ts": float(ts) if emb is not None else 0.0,
                    "emb": emb,
                    "zone": zone or "default",
                }
                self.active_tracks[key] = rec
                # Track zones and gallery
                self.active_zones[int(sid)].add((int(sensor_id), rec["zone"]))
                if emb is not None:
                    sid_int = int(sid)
                    self.gallery[sid_int].append(emb)
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
                return int(sid)

            # Existing track: update
            rec["bbox"] = bbox_ltrbwh
            rec["last_seen_ts"] = float(ts)
            if zone:
                rec["zone"] = zone
                self.active_zones[int(rec["stable_id"])].add((int(sensor_id), rec["zone"]))
            if emb is not None:
                rec["emb"] = emb
                rec["last_emb_ts"] = float(ts)
                sid_int = int(rec["stable_id"]) 
                self.gallery[sid_int].append(emb)
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
            self.active_tracks[key] = rec
            try:
                self.sid_global_last_seen[int(rec["stable_id"])] = float(ts)
            except Exception:
                pass
            return int(rec["stable_id"])

    def remove_missing_tracks(self, sensor_id: int, present_ds_ids: List[int], ts: float) -> None:
        """Move tracks not present this frame to ghost lists and update active_zones.

        Called at end of a frame for a specific sensor.
        """
        with self._lock:
            present_set = set(int(i) for i in present_ds_ids)
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

                # Push to ghost list for this camera if we have an embedding
                if rec.get("emb", None) is not None:
                    self.ghosts[int(sensor_id)].append(
                        {
                            "stable_id": sid,
                            "bbox": rec.get("bbox"),
                            "ts": float(ts),
                            "emb": rec.get("emb"),
                        }
                    )

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
                    if (t - float(last_seen)) >= max(2.0, self.active_evict_grace_s):
                        self._purge_sid_state(sid)
                        self._free_sid(sid)
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
                return {
                    "active_unique": len(active_sids),
                    "active_by_sensor": active_counts,
                    "ghost_unique": len(ghost_sids),
                    "ghost_entries": total_ghosts,
                    "gallery_ids": len(self.gallery),
                    "free_sid_pool_size": free_pool_size,
                    "next_sid": int(self.next_stable_id),
                    "pending_new_count": int(pending_new),
                }
            except Exception:
                return {}

    # --------------- Internal ---------------
    def _match_ghost(self, sensor_id: int, emb: np.ndarray, bbox: BBox, ts: float) -> Optional[int]:
        dq = self.ghosts.get(int(sensor_id))
        if not dq:
            return None
        # Loose spatial check threshold based on bbox diagonal
        x, y, w, h = bbox
        diag = float((w ** 2 + h ** 2) ** 0.5)
        best_sid, best_sim = None, -1.0
        for ghost in reversed(dq):  # newest first
            if float(ts) - float(ghost.get("ts", 0.0)) > self.max_ghost_age_s:
                continue
            g_sid = int(ghost.get("stable_id"))
            # Exclusivity only if multi-active not allowed
            if not self.allow_multi_zone_active and self.active_zones.get(g_sid):
                continue
            gx, gy, gw, gh = ghost.get("bbox", (0, 0, 0, 0))
            dist = float(((x - gx) ** 2 + (y - gy) ** 2) ** 0.5)
            if dist > 1.5 * diag:
                continue
            g_emb = ghost.get("emb")
            if g_emb is None:
                continue
            sim = self._cosine(emb, g_emb)
            # Age-adaptive threshold: require extra margin for older ghosts
            age = float(ts) - float(ghost.get("ts", 0.0))
            thr = self.cos_sim_threshold + (self.ghost_extra_margin if age >= self.ghost_strict_age_s else 0.0)
            if sim >= thr and sim > best_sim:
                best_sid, best_sim = g_sid, sim
        return best_sid
