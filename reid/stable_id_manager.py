import threading
import time
import math
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Mapping, Optional, Set, Tuple, Any, Iterable
import heapq
import os
import json

import numpy as np

from noesis_core.strict_json import strict_json_loads

from .assignment import (
    FrameAssignmentState,
    active_embeddings_same_frame,
    is_mutual_nearest_match,
)
from .camera_topology import CameraTopology, load_camera_topology
from .embedding_extractor import EmbeddingExtractor
from .gallery_quality import add_clustered_exemplar, should_accept_gallery_embedding
from .household_identity import (
    PROVISIONAL_ID_MIN,
    VISITOR_ID_MAX,
    VISITOR_ID_MIN,
    ProvisionalPool,
    ResidentRegistry,
    VisitorPool,
    PRIVATE_FILE_MODE,
    ensure_private_directory,
    secure_private_file,
    atomic_write_json_private,
    copy_gallery_embeddings,
    default_household_paths,
    identity_kind_for_sid,
    identity_state_for_kind,
)


BBox = Tuple[float, float, float, float]  # left, top, width, height
logger = logging.getLogger(__name__)


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
        # Household identity mode (Phase 0): exclusivity + overlap permits.
        household_mode: bool = False,
        camera_topology_file: Optional[str] = None,
        camera_labels: Optional[Mapping[int, str]] = None,
        overlap_allow_appearance_only: bool = False,
        # Household closed-world identity (Phase 1)
        residents_file: Optional[str] = None,
        visitor_pool_file: Optional[str] = None,
        visitor_id_min: int = VISITOR_ID_MIN,
        visitor_id_max: int = VISITOR_ID_MAX,
        visitor_ttl_s: float = 3600.0,
        household_confirm_embeddings: int = 3,
        resident_match_margin: float = 0.06,
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
        # Cross-camera handoff: relax the gallery-match requirement for
        # identities seen recently on any camera (walking between rooms).
        xcam_handoff_window_s: float = 12.0,
        xcam_handoff_margin: float = 0.06,
        # Early reconcile: a fresh low-support track keeps re-testing ghost and
        # gallery matches as better embeddings arrive, so an ID minted from a
        # weak far-away first crop can still converge to the person's real SID.
        early_reconcile_window_s: float = 6.0,
        early_reconcile_max_attempts: int = 4,
        early_reconcile_max_support: int = 4,
        # Global ID pool soft-cap
        max_total_ids: int = 12,
        total_id_reuse: bool = True,
        total_id_reuse_min_age_s: float = 600.0,
        # Alias-auto merge controls.
        auto_merge_enabled: bool = False,
        auto_merge_interval_s: float = 30.0,
        auto_merge_max_attempts: int = 5,
        auto_merge_min_sim: float = 0.92,
        auto_merge_min_embeddings_for_suggest: int = 1,
        auto_merge_respect_inactive: bool = True,
        auto_merge_force_stuck: bool = False,
        auto_merge_allow_both_active: bool = False,
        auto_merge_both_active_min_sim: float = 0.97,
        # New-ID hysteresis (global, even when not at cap)
        new_id_hysteresis_frames: int = 2,
        # SID allocator persistence (smarter restart)
        sid_pool_file: str = "~/.noesis/sid_pool.json",
        reset_sid_pool_on_start: bool = False,
        # Long-term identity memory: persist gallery embeddings across restarts
        # so people can be re-identified after long absences. Disabled when None.
        gallery_persist_file: Optional[str] = None,
        gallery_persist_max_age_s: float = 604800.0,
        gallery_autosave_interval_s: float = 30.0,
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
        self.household_mode = bool(household_mode)
        self.overlap_allow_appearance_only = bool(overlap_allow_appearance_only)
        if self.household_mode:
            allow_multi_zone_active = False
            auto_merge_enabled = False
            total_id_reuse = False
            gallery_size = min(int(gallery_size), 8)
        self.allow_multi_zone_active = bool(allow_multi_zone_active)
        self._camera_topology: CameraTopology = load_camera_topology(camera_topology_file)
        self._camera_topology_validation_status = "not_validated"
        self._camera_topology_validation_errors: List[str] = []
        if camera_labels is not None:
            errors = self._camera_topology.validate_runtime_map(camera_labels)
            if self._camera_topology.source_path is None:
                errors.append("camera topology file is not loaded")
            self._camera_topology_validation_errors = list(errors)
            self._camera_topology_validation_status = "invalid" if errors else "valid"
            if self.household_mode and errors:
                raise ValueError("camera topology/runtime map mismatch: " + "; ".join(errors))
        if self.household_mode and self._camera_topology.overlap_allow_appearance_only:
            self.overlap_allow_appearance_only = True
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
        # Allow limited same-sensor reassociation when geometry strongly supports continuity.
        self.active_id_guard_relaxed_reid_slack = 0.05
        self.active_id_guard_relaxed_dist_ratio = 1.25
        self.active_id_guard_relaxed_scale_ratio = 2.50
        # Hard uniqueness guard: do not allow duplicate SID claims within one
        # sensor frame (same timestamp).
        self.same_frame_sid_epsilon_s = 1e-2
        self.ghost_strict_age_s = float(ghost_strict_age_s)
        self.ghost_extra_margin = float(ghost_extra_margin)
        self.max_active_ids_per_sensor = int(max_active_ids_per_sensor)
        self.new_id_confirm_frames_at_cap = int(new_id_confirm_frames_at_cap)
        self.active_evict_grace_s = float(active_evict_grace_s)
        self.xcam_handoff_window_s = float(xcam_handoff_window_s)
        self.xcam_handoff_margin = float(xcam_handoff_margin)
        self.early_reconcile_window_s = float(early_reconcile_window_s)
        self.early_reconcile_max_attempts = int(max(0, early_reconcile_max_attempts))
        self.early_reconcile_max_support = int(max(1, early_reconcile_max_support))
        self.max_total_ids = int(max_total_ids)
        self.total_id_reuse = bool(total_id_reuse)
        self.total_id_reuse_min_age_s = float(total_id_reuse_min_age_s)
        self.auto_merge_enabled = bool(auto_merge_enabled)
        self.auto_merge_interval_s = float(auto_merge_interval_s)
        self.auto_merge_max_attempts = int(max(1, auto_merge_max_attempts))
        self.auto_merge_min_sim = float(auto_merge_min_sim)
        self.auto_merge_min_embeddings_for_suggest = int(max(1, auto_merge_min_embeddings_for_suggest))
        self.auto_merge_respect_inactive = bool(auto_merge_respect_inactive)
        self.auto_merge_force_stuck = bool(auto_merge_force_stuck)
        self.auto_merge_allow_both_active = bool(auto_merge_allow_both_active)
        self.auto_merge_both_active_min_sim = float(auto_merge_both_active_min_sim)
        self.new_id_hysteresis_frames = int(new_id_hysteresis_frames)
        self.sid_pool_file = os.path.expanduser(str(sid_pool_file))
        self.gallery_persist_file = (
            os.path.expanduser(str(gallery_persist_file)) if gallery_persist_file else None
        )
        self.gallery_persist_max_age_s = float(gallery_persist_max_age_s)
        self.gallery_autosave_interval_s = float(gallery_autosave_interval_s)
        self._gallery_last_save_ts = 0.0
        self._gallery_loaded_sids = 0
        # Autosave persistence must never make the frame callback wait for
        # compression or filesystem I/O. The callback snapshots the small
        # in-memory gallery under ``_lock``; a bounded daemon writer performs
        # the atomic file replacement independently. Explicit ``save_gallery``
        # calls (including shutdown) remain synchronous.
        self._gallery_autosave_lock = threading.Lock()
        self._gallery_autosave_thread: Optional[threading.Thread] = None
        self._gallery_file_write_lock = threading.Lock()
        # Snapshot order is independent of filesystem-lock acquisition order.
        # An explicit save can therefore win the file lock before an older
        # autosave writer and prevent that older snapshot from regressing the
        # durable gallery.
        self._gallery_save_order_lock = threading.Lock()
        self._gallery_save_generation = 0
        self._gallery_last_written_generation = 0
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
        if self.household_mode:
            self.gallery_size = int(min(self.gallery_size, 8))
            self.gpu_min_gallery = int(min(self.gpu_min_gallery, 8))
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
        # Sensor that produced sid_last_bbox; scale/spatial penalties only make
        # sense within the same camera view (cross-camera size differences are
        # expected geometry, not identity evidence).
        self.sid_last_bbox_sensor: Dict[int, int] = {}
        self.sid_last_brightness: Dict[int, float] = {}
        self.sid_last_color: Dict[int, np.ndarray] = {}
        # Global last seen timestamp per stable_id (any sensor)
        self.sid_global_last_seen: Dict[int, float] = {}
        # Global first seen timestamp per stable_id
        self.sid_global_first_seen: Dict[int, float] = {}
        # Alias map: src_sid -> dst_sid (may chain)
        self.sid_alias: Dict[int, int] = {}
        # Alias source IDs are allocator-reserved to avoid recycling IDs that
        # would immediately canonicalize into another SID.
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
        # Track-level diagnostics for telemetry.
        self._pending_id_diag: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Identity churn / decision counters.
        self._sid_new_alloc_count = 0
        self._sid_remap_count = 0
        self._sid_guard_reject_count = 0
        self._sid_no_embedding_count = 0
        self._sid_fragmentation_events = 0
        self._sid_pending_recycled_count = 0
        self._sid_same_frame_conflict_count = 0
        self._false_share_blocked_count = 0
        self._overlap_permit_grant_count = 0
        self._overlap_permit_deny_count = 0
        self._overlap_permit_degraded_count = 0
        self._gallery_quality_reject_count = 0
        self._mnn_reject_count = 0
        self._frame_assignment = FrameAssignmentState(epsilon_s=self.same_frame_sid_epsilon_s)
        # Household closed-world state (Phase 1)
        self._resident_registry: Optional[ResidentRegistry] = None
        self._visitor_pool: Optional[VisitorPool] = None
        self._provisional_pool: Optional[ProvisionalPool] = None
        self.visitor_id_min = int(visitor_id_min)
        self.visitor_id_max = int(visitor_id_max)
        self.visitor_ttl_s = float(visitor_ttl_s)
        self.household_confirm_embeddings = int(max(1, household_confirm_embeddings))
        self.resident_match_margin = float(resident_match_margin)
        self._household_emb_counts: Dict[Tuple[int, int], int] = {}
        self._mint_visitor_count = 0
        self._promote_resident_count = 0
        # Cumulative provisional mint events (first alloc per track lifecycle).
        self._provisional_event_count = 0
        if self.household_mode:
            hh_paths = default_household_paths()
            residents_path = residents_file or hh_paths["residents_file"]
            visitor_pool_path = visitor_pool_file or hh_paths["visitor_pool_file"]
            for state_path in (
                residents_path,
                visitor_pool_path,
                self.gallery_persist_file,
                self.alias_file,
                self.sid_pool_file,
            ):
                if not state_path:
                    continue
                parent = os.path.dirname(os.path.expanduser(str(state_path)))
                if parent:
                    ensure_private_directory(parent)
            self._resident_registry = ResidentRegistry(str(residents_path))
            self._visitor_pool = VisitorPool(
                pool_file=str(visitor_pool_path),
                id_min=self.visitor_id_min,
                id_max=self.visitor_id_max,
                ttl_s=self.visitor_ttl_s,
            )
            self._provisional_pool = ProvisionalPool()
        # Last world/footpoint per (stable_id, sensor_id) for overlap geometry.
        self._sid_sensor_world: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._id_event_counts: Dict[str, int] = defaultdict(int)
        # Alias auto-merge accounting.
        self._last_auto_merge_ts: Optional[float] = None
        self._auto_merge_runs = 0
        self._auto_merge_triggers = 0
        self._auto_merge_applied = 0
        self._auto_merge_suppressed = 0
        self._auto_merge_last_candidate_count = 0
        self._auto_merge_last_blocked_count = 0
        self._auto_merge_last_support_gate = int(self.auto_merge_min_embeddings_for_suggest)
        self._auto_merge_zero_apply_streak = 0
        self._auto_merge_last_warn_ts: Optional[float] = None
        self._auto_merge_last_reason = "idle"
        self._auto_merge_last_pressure_recycled = 0
        self._auto_merge_pressure_recycled_total = 0
        self._alias_candidate_pool_size = 0
        self._alias_candidate_sim_p50: Optional[float] = None
        self._alias_candidate_sim_p95: Optional[float] = None
        self._alias_support_ids_ge_1 = 0
        self._alias_support_ids_ge_2 = 0
        self._alias_support_ids_ge_3 = 0
        self.reset_sid_pool_on_start = bool(reset_sid_pool_on_start)

        self.next_stable_id = 1
        if self.household_mode:
            self.next_stable_id = int(self.visitor_id_min)
        # Free-list allocator state
        self._free_sids: List[int] = []
        self._free_sids_set: set[int] = set()
        if self.aliases_enabled:
            self._load_aliases()
            self._refresh_alias_reserved()
        if self.reset_sid_pool_on_start:
            self._free_sids = []
            self._free_sids_set = set()
            self._clear_sid_pool_file()
        self._load_sid_pool()
        if self.aliases_enabled and self.sid_alias_reserved:
            self._free_sids = [sid for sid in self._free_sids if sid not in self.sid_alias_reserved]
            heapq.heapify(self._free_sids)
            self._free_sids_set = set(self._free_sids)
        self._load_gallery()
        if self.household_mode:
            self._reconcile_household_pool_state_after_load()
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
        # Force the first production-shaped similarity kernel and device-host
        # synchronization during startup. Without this, Torch/CUDA lazy
        # initialization lands inside the first frame that reaches a gallery
        # match (observed as a one-time ~200 ms StableID callback tail).
        try:
            with torch.no_grad():
                warm_query = torch.as_tensor(
                    np.zeros((256,), dtype=np.float32),
                    device=dev,
                )
                warm_gallery = torch.as_tensor(
                    np.zeros((256, 256), dtype=np.float32),
                    device=dev,
                )
                warm_query = warm_query / (torch.norm(warm_query) + 1e-12)
                warm_gallery = warm_gallery / (
                    torch.linalg.norm(warm_gallery, dim=1, keepdim=True) + 1e-12
                )
                torch.matmul(warm_gallery, warm_query).detach().cpu().numpy()
        except Exception as exc:
            self._backend_last_error = f"gpu_warmup:{type(exc).__name__}"
            if strict_gpu:
                raise RuntimeError(
                    f"StableID GPU similarity warmup failed: {type(exc).__name__}"
                ) from exc

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
        """Per-SID similarity as max over the EMA centroid and gallery exemplars.

        Matching against stored exemplars (not just the recency-weighted
        centroid) lets an identity be recalled from any previously seen
        appearance (pose, lighting, partial view), which is what makes
        long-absence re-identification work.
        """
        sims: Dict[int, float] = {}
        if emb is None or len(candidates) < 1:
            return sims
        emb_vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        rows: List[np.ndarray] = []
        row_sids: List[int] = []
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
            added = False
            if centroid is not None:
                c = np.asarray(centroid, dtype=np.float32).reshape(-1)
                if c.shape[0] == emb_vec.shape[0]:
                    rows.append(c)
                    row_sids.append(sid_int)
                    added = True
            for _ts, v in self.gallery.get(sid_int, ()):  # exemplars
                if v is None:
                    continue
                vv = np.asarray(v, dtype=np.float32).reshape(-1)
                if vv.shape[0] != emb_vec.shape[0]:
                    continue
                rows.append(vv)
                row_sids.append(sid_int)
                added = True
            if not added:
                continue
        if not rows:
            return sims

        use_gpu = bool(self._gpu_enabled and self._backend_mode == "gpu" and len(rows) >= int(self.gpu_min_gallery))
        if use_gpu:
            try:
                torch = self._torch
                if torch is None or self._torch_device is None:
                    raise RuntimeError("torch_backend_uninitialized")
                q = torch.as_tensor(emb_vec, dtype=torch.float32, device=self._torch_device)
                q = q / (torch.norm(q) + 1e-12)
                mat = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=self._torch_device)
                mat = mat / (torch.linalg.norm(mat, dim=1, keepdim=True) + 1e-12)
                sim_vec = torch.matmul(mat, q)
                sim_np = sim_vec.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception as exc:
                self._backend_last_error = f"gpu_similarity:{type(exc).__name__}"
                raise RuntimeError(f"StableID GPU similarity failed: {type(exc).__name__}") from exc
        else:
            mat_np = np.stack(rows, axis=0).astype(np.float32, copy=False)
            sim_np = np.matmul(mat_np, emb_vec.reshape(-1, 1)).reshape(-1)

        for idx, sid_int in enumerate(row_sids):
            val = float(sim_np[idx])
            prev = sims.get(int(sid_int))
            if prev is None or val > prev:
                sims[int(sid_int)] = val
        return sims

    # --------------- Household closed-world (Phase 1) -----------------
    def _household_resident_ids(self) -> set:
        if self._resident_registry is None:
            return set()
        return self._resident_registry.resident_ids()

    def _reconcile_household_pool_state_after_load(self) -> None:
        """Reserve persisted visitor-gallery slots after a restart.

        Older visitor-pool files can be absent or incomplete even though the
        gallery still contains a visitor. Reserving those slots prevents a new
        visitor from being assigned the same numeric SID in the same process.
        """
        if self._visitor_pool is None:
            return
        before = self._visitor_pool.used_sids()
        for sid in list(self.gallery.keys()):
            sid_int = int(sid)
            if not (self.visitor_id_min <= sid_int <= self.visitor_id_max):
                continue
            last_seen = float(self.sid_global_last_seen.get(sid_int, time.time()))
            self._visitor_pool.touch(sid_int, last_seen)
        if self._visitor_pool.used_sids() != before:
            self._visitor_pool.save()

    def _household_identity_kind(self, sid: int) -> str:
        return identity_kind_for_sid(int(sid), resident_ids=self._household_resident_ids())

    def _household_identity_state(self, rec: Optional[Dict[str, Any]]) -> str:
        if rec is None:
            return "provisional"
        kind = str(rec.get("identity_kind") or self._household_identity_kind(int(rec.get("stable_id", 0))))
        return identity_state_for_kind(kind)

    def _household_is_provisional_sid(self, sid: int) -> bool:
        if self._provisional_pool is not None and self._provisional_pool.is_provisional(int(sid)):
            return True
        return int(sid) >= PROVISIONAL_ID_MIN

    def _household_should_persist_gallery(self, rec: Dict[str, Any]) -> bool:
        kind = str(rec.get("identity_kind") or self._household_identity_kind(int(rec.get("stable_id", 0))))
        return kind in ("resident", "visitor")

    def _household_live_provisional_sids(self) -> Set[int]:
        live: Set[int] = set()
        for rec in self.active_tracks.values():
            try:
                sid = int(rec.get("stable_id", -1))
            except Exception:
                continue
            if self._household_is_provisional_sid(sid):
                live.add(sid)
        for pending in self._pending_new_sids.values():
            try:
                sid = int(pending)
            except Exception:
                continue
            if self._household_is_provisional_sid(sid):
                live.add(sid)
        return live

    def _household_reclaim_orphan_provisionals(self) -> int:
        if self._provisional_pool is None:
            return 0
        recycled = self._provisional_pool.reclaim_orphans(self._household_live_provisional_sids())
        return int(len(recycled))

    def _household_evict_oldest_provisional(self, *, exclude_key: Optional[Tuple[int, int]] = None) -> Optional[int]:
        """Drop the oldest provisional active track and free its SID for reuse."""
        oldest_key: Optional[Tuple[int, int]] = None
        oldest_ts = float("inf")
        oldest_sid: Optional[int] = None
        for key, rec in self.active_tracks.items():
            if exclude_key is not None and key == exclude_key:
                continue
            kind = str(rec.get("identity_kind") or "")
            try:
                sid = int(rec.get("stable_id", -1))
            except Exception:
                continue
            if kind != "provisional" and not self._household_is_provisional_sid(sid):
                continue
            try:
                last_ts = float(rec.get("last_seen_ts", rec.get("first_seen_ts", 0.0)))
            except Exception:
                last_ts = 0.0
            if last_ts < oldest_ts:
                oldest_ts = last_ts
                oldest_key = key
                oldest_sid = sid
        if oldest_key is None or oldest_sid is None:
            return None
        self.active_tracks.pop(oldest_key, None)
        self._pending_id_diag.pop(oldest_key, None)
        self._pending_new_counts.pop(oldest_key, None)
        self._pending_new_ts.pop(oldest_key, None)
        self._pending_new_sids.pop(oldest_key, None)
        self._household_emb_counts.pop(oldest_key, None)
        zone_pairs = self.active_zones.get(int(oldest_sid), set())
        if zone_pairs:
            try:
                sensor_int = int(oldest_key[0])
                zone_pairs = {pair for pair in zone_pairs if int(pair[0]) != sensor_int}
            except Exception:
                zone_pairs = set()
            if zone_pairs:
                self.active_zones[int(oldest_sid)] = zone_pairs
            else:
                self.active_zones.pop(int(oldest_sid), None)
        still_used = any(
            int(r.get("stable_id", -1)) == int(oldest_sid) for r in self.active_tracks.values()
        )
        if not still_used:
            self._household_release_provisional(int(oldest_sid))
        return int(oldest_sid)

    def _household_alloc_provisional(self, *, exclude_key: Optional[Tuple[int, int]] = None) -> int:
        if self._provisional_pool is None:
            return int(self._alloc_sid())
        # Safety net: release provisionals that are no longer live.
        self._household_reclaim_orphan_provisionals()
        try:
            return int(self._provisional_pool.alloc())
        except RuntimeError:
            # Under tracker churn / false detections, evict oldest provisional and retry.
            for _ in range(3):
                freed = self._household_evict_oldest_provisional(exclude_key=exclude_key)
                if freed is None:
                    break
                try:
                    return int(self._provisional_pool.alloc())
                except RuntimeError:
                    continue
            self._household_reclaim_orphan_provisionals()
            return int(self._provisional_pool.alloc())

    def _household_release_provisional(self, sid: Optional[int]) -> None:
        if sid is None or self._provisional_pool is None:
            return
        if self._household_is_provisional_sid(int(sid)):
            self._provisional_pool.release(int(sid))

    def _household_pressure_recycle_visitors(self, ts: float) -> int:
        """Recycle inactive visitors immediately when the free pool is empty.

        Under pressure, ghost entries for inactive visitors are dropped so the
        closed visitor range can rotate instead of blocking forever.
        """
        if self._visitor_pool is None:
            return 0
        active_sids = {int(rec.get("stable_id")) for rec in self.active_tracks.values()}
        recycled = 0
        # Drop ghost claims for visitors that are not currently active.
        for sensor_id, dq in list(self.ghosts.items()):
            if not dq:
                continue
            kept = []
            for ghost in dq:
                try:
                    g_sid = int(ghost.get("stable_id"))
                except Exception:
                    continue
                if (
                    self.visitor_id_min <= g_sid <= self.visitor_id_max
                    and g_sid not in active_sids
                ):
                    continue
                kept.append(ghost)
            self.ghosts[int(sensor_id)].clear()
            self.ghosts[int(sensor_id)].extend(kept)
        for sid, _last in list(self._visitor_pool._last_seen.items()):
            sid_int = int(sid)
            if sid_int in active_sids:
                continue
            self._visitor_pool.release(sid_int)
            self._purge_sid_state(sid_int)
            recycled += 1
        if recycled:
            try:
                self._visitor_pool.save()
            except Exception:
                pass
        return int(recycled)

    def _household_mint_visitor(self, ts: float) -> int:
        if self._visitor_pool is None:
            raise RuntimeError("visitor pool unavailable in household mode")
        try:
            sid = int(self._visitor_pool.alloc(float(ts)))
        except RuntimeError:
            self._household_pressure_recycle_visitors(float(ts))
            sid = int(self._visitor_pool.alloc(float(ts)))
        self._mint_visitor_count += 1
        self.next_stable_id = max(int(self.next_stable_id), int(sid) + 1)
        return sid

    def _household_touch_visitor(self, sid: int, ts: float) -> None:
        if self._visitor_pool is not None and self.visitor_id_min <= int(sid) <= self.visitor_id_max:
            self._visitor_pool.touch(int(sid), float(ts))

    def _household_record_embedding_confirm(self, key: Tuple[int, int], emb: Optional[np.ndarray]) -> int:
        if emb is None:
            return int(self._household_emb_counts.get(key, 0))
        count = int(self._household_emb_counts.get(key, 0)) + 1
        self._household_emb_counts[key] = count
        return count

    def _household_confirmed(self, key: Tuple[int, int], emb: Optional[np.ndarray]) -> bool:
        emb_count = self._household_record_embedding_confirm(key, emb)
        pending_frames = int(self._pending_new_counts.get(key, 0))
        frame_ok = pending_frames >= max(1, int(self.new_id_hysteresis_frames))
        emb_ok = emb is not None and emb_count >= int(self.household_confirm_embeddings)
        return bool(frame_ok and emb_ok)

    def _household_gallery_best(
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
    ) -> Tuple[Optional[int], float, float, str]:
        """Resident-first gallery match with stricter resident threshold."""
        resident_ids = sorted(self._household_resident_ids())
        visitor_ids = sorted(
            sid
            for sid in self.gallery.keys()
            if self.visitor_id_min <= int(sid) <= self.visitor_id_max
        )
        resident_req = float(self.cos_sim_high_threshold) + float(self.resident_match_margin)
        if min_reid is not None:
            resident_req = max(float(min_reid), resident_req)

        best: Tuple[Optional[int], float, float, str] = (None, -1.0, resident_req, "visitor")
        if resident_ids:
            sims = self._similarity_for_candidates(emb, resident_ids)
            for sid in resident_ids:
                sim = sims.get(int(sid))
                if sim is None or float(sim) < resident_req:
                    continue
                if float(sim) > best[1]:
                    best = (int(sid), float(sim), float(resident_req), "resident")

        if best[0] is not None:
            return best

        g_id, g_reid, g_req = self._gallery_best(
            emb,
            sensor_id=sensor_id,
            curr_bbox=curr_bbox,
            curr_brightness=curr_brightness,
            curr_color=curr_color,
            pose_vec=pose_vec,
            pose_valid=pose_valid,
            now_ts=now_ts,
            min_reid=min_reid,
        )
        kind = "visitor"
        if g_id is not None and int(g_id) in self._household_resident_ids():
            kind = "resident"
        elif g_id is not None and not (self.visitor_id_min <= int(g_id) <= self.visitor_id_max):
            kind = "visitor"
        return g_id, float(g_reid), float(g_req), kind

    def _household_apply_identity_meta(
        self,
        rec: Dict[str, Any],
        *,
        sid: int,
        kind: Optional[str] = None,
    ) -> None:
        sid_int = int(sid)
        identity_kind = str(kind or self._household_identity_kind(sid_int))
        rec["stable_id"] = sid_int
        rec["identity_kind"] = identity_kind
        rec["identity_state"] = identity_state_for_kind(identity_kind)
        if self._resident_registry is not None and identity_kind == "resident":
            rec["resident_uuid"] = self._resident_registry.uuid_for(sid_int)
            rec["display_name"] = self._resident_registry.display_name_for(sid_int)
        else:
            rec["resident_uuid"] = None
            rec["display_name"] = None
        if self._visitor_pool is not None and identity_kind == "visitor":
            rec["visitor_generation"] = self._visitor_pool.generation_for(sid_int)
        else:
            rec.pop("visitor_generation", None)

    def _count_active_provisional(self) -> int:
        count = 0
        for rec in self.active_tracks.values():
            try:
                kind = str(rec.get("identity_kind") or "")
                sid = int(rec.get("stable_id", -1))
            except Exception:
                continue
            if kind == "provisional" or self._household_is_provisional_sid(sid):
                count += 1
        return int(count)

    def _remap_sid(
        self,
        old_sid: int,
        new_sid: int,
        *,
        kind: Optional[str] = None,
        purge_old: bool = False,
        move_gallery: bool = True,
    ) -> None:
        """Remap active tracks/zones/ghosts (and optionally gallery) from old→new SID."""
        old_int = int(old_sid)
        new_int = int(new_sid)
        identity_kind = str(kind) if kind is not None else None

        if old_int == new_int:
            if identity_kind is not None:
                for track in self.active_tracks.values():
                    try:
                        if int(track.get("stable_id", -1)) == new_int:
                            self._household_apply_identity_meta(track, sid=new_int, kind=identity_kind)
                    except Exception:
                        continue
            return

        for track in self.active_tracks.values():
            try:
                if int(track.get("stable_id", -1)) != old_int:
                    continue
            except Exception:
                continue
            if identity_kind is not None:
                self._household_apply_identity_meta(track, sid=new_int, kind=identity_kind)
            else:
                track["stable_id"] = new_int
                if self.household_mode:
                    self._household_apply_identity_meta(track, sid=new_int)

        old_pairs = self.active_zones.pop(old_int, set())
        if old_pairs:
            self.active_zones[new_int].update(old_pairs)

        for dq in self.ghosts.values():
            for ghost in dq:
                try:
                    if int(ghost.get("stable_id", -1)) == old_int:
                        ghost["stable_id"] = new_int
                except Exception:
                    continue

        if move_gallery:
            copy_gallery_embeddings(self.gallery, old_int, new_int)
            src_pose = list(self.pose_gallery.get(old_int, []))
            if src_pose:
                dst_pose = list(self.pose_gallery.get(new_int, []))
                maxlen = self.pose_gallery[new_int].maxlen or self.pose_gallery_size
                combined_pose = dst_pose + src_pose
                combined_pose.sort(key=lambda item: float(item[0]), reverse=True)
                self.pose_gallery[new_int] = deque(combined_pose[:maxlen], maxlen=maxlen)
                pose_vecs = [v for (_ts, v) in self.pose_gallery[new_int] if v is not None]
                if pose_vecs:
                    centroid = np.mean(np.stack(pose_vecs, axis=0), axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                    self.pose_centroid[new_int] = centroid.astype(np.float32)
            if old_int in self.pose_last_seen or new_int in self.pose_last_seen:
                self.pose_last_seen[new_int] = max(
                    float(self.pose_last_seen.get(old_int, 0.0)),
                    float(self.pose_last_seen.get(new_int, 0.0)),
                )
            if old_int in self.sid_centroid:
                if new_int not in self.sid_centroid:
                    self.sid_centroid[new_int] = self.sid_centroid[old_int]
                else:
                    self._recompute_sid_centroid(new_int)
            for attr in (
                "sid_last_bbox",
                "sid_last_bbox_sensor",
                "sid_last_brightness",
                "sid_last_color",
            ):
                store = getattr(self, attr, None)
                if not isinstance(store, dict):
                    continue
                if old_int in store and new_int not in store:
                    store[new_int] = store[old_int]
            last_old = self.sid_global_last_seen.get(old_int)
            last_new = self.sid_global_last_seen.get(new_int)
            if last_old is not None or last_new is not None:
                if last_old is None:
                    self.sid_global_last_seen[new_int] = float(last_new)  # type: ignore[arg-type]
                elif last_new is None:
                    self.sid_global_last_seen[new_int] = float(last_old)
                else:
                    self.sid_global_last_seen[new_int] = float(max(last_old, last_new))
            first_old = self.sid_global_first_seen.get(old_int)
            first_new = self.sid_global_first_seen.get(new_int)
            if first_old is not None or first_new is not None:
                candidates = [float(x) for x in (first_old, first_new) if x is not None]
                if candidates:
                    self.sid_global_first_seen[new_int] = float(min(candidates))
            for key in list(self._sid_sensor_world.keys()):
                try:
                    sid_k, sensor_k = key
                except Exception:
                    continue
                if int(sid_k) != old_int:
                    continue
                payload = self._sid_sensor_world.pop(key, None)
                if payload is not None:
                    self._sid_sensor_world[(new_int, int(sensor_k))] = payload

        if purge_old:
            self.gallery.pop(old_int, None)
            self.sid_centroid.pop(old_int, None)
            self.sid_last_bbox.pop(old_int, None)
            self.sid_last_bbox_sensor.pop(old_int, None)
            self.sid_last_brightness.pop(old_int, None)
            self.sid_last_color.pop(old_int, None)
            self.active_zones.pop(old_int, None)
            self.pose_gallery.pop(old_int, None)
            self.pose_centroid.pop(old_int, None)
            self.pose_last_seen.pop(old_int, None)
            self.sid_global_last_seen.pop(old_int, None)
            self.sid_global_first_seen.pop(old_int, None)
            for key in list(self._sid_sensor_world.keys()):
                try:
                    if int(key[0]) == old_int:
                        self._sid_sensor_world.pop(key, None)
                except Exception:
                    continue
            for pair in list(self.sid_last_copresent.keys()):
                try:
                    if int(pair[0]) == old_int or int(pair[1]) == old_int:
                        self.sid_last_copresent.pop(pair, None)
                except Exception:
                    self.sid_last_copresent.pop(pair, None)
            if self._household_is_provisional_sid(old_int):
                self._household_release_provisional(old_int)
            elif self._visitor_pool is not None and self.visitor_id_min <= old_int <= self.visitor_id_max:
                self._visitor_pool.release(old_int)

        self._sid_remap_count += 1

    def _household_finalize_new_sid(
        self,
        key: Tuple[int, int],
        *,
        emb: Optional[np.ndarray],
        sensor_id: int,
        bbox_ltrbwh: BBox,
        ts: float,
        curr_brightness: Optional[float],
        curr_color: Optional[np.ndarray],
        pose_vec: Optional[np.ndarray],
        pose_valid: bool,
        world_xy: Optional[Tuple[float, float]],
        world_valid: bool,
        pending_sid: Optional[int],
    ) -> Tuple[int, str, str]:
        """Match resident/visitor or mint visitor after confirmation gates."""
        diag_event = "new_alloc"
        sid: Optional[int] = None
        identity_kind = "visitor"

        if emb is not None:
            g_id, g_reid, g_req, kind = self._household_gallery_best(
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
                ok, _reason = self._gallery_match_ok(
                    track_key=key,
                    emb=emb,
                    g_id=int(g_id),
                    g_reid=float(g_reid),
                    g_req=float(g_req),
                    sensor_id=int(sensor_id),
                    bbox=bbox_ltrbwh,
                    ts=float(ts),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                )
                if ok:
                    sid = int(g_id)
                    identity_kind = str(kind)
                    diag_event = "match_gallery"

        if sid is None:
            sid = int(self._household_mint_visitor(float(ts)))
            identity_kind = "visitor"
            diag_event = "mint_visitor"

        return int(sid), str(identity_kind), str(diag_event)

    def list_residents(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._resident_registry is None:
                return []
            out: List[Dict[str, Any]] = []
            for row in self._resident_registry.list_residents():
                sid = int(row.get("stable_id", 0))
                emb_count = int(len(self.gallery.get(sid, [])))
                payload = dict(row)
                # The registry's count is a cached persistence hint. The live
                # gallery is authoritative and can shrink/grow after enrollment.
                payload["embedding_count"] = emb_count
                payload["gallery_embeddings"] = emb_count
                out.append(payload)
            return out

    def enroll_resident(
        self,
        *,
        display_name: str,
        stable_id: Optional[int] = None,
        visitor_id: Optional[int] = None,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(now_ts if now_ts is not None else time.time())
        with self._lock:
            if self._resident_registry is None:
                raise RuntimeError("household mode not enabled")
            bind_sid = stable_id if stable_id is not None else visitor_id
            source_sid: Optional[int] = None
            if visitor_id is not None:
                source_sid = int(visitor_id)
            elif bind_sid is not None:
                source_sid = int(bind_sid)
            emb_count = 0
            if source_sid is not None:
                emb_count = int(len(self.gallery.get(int(source_sid), [])))
            rec = self._resident_registry.enroll(
                display_name=str(display_name),
                stable_id=stable_id,
                visitor_id=visitor_id,
                embedding_count=emb_count,
                now_ts=now,
            )
            new_sid = int(rec.stable_id)
            if source_sid is not None and int(source_sid) != new_sid:
                self._remap_sid(
                    int(source_sid),
                    new_sid,
                    kind="resident",
                    purge_old=True,
                    move_gallery=True,
                )
            else:
                for track in self.active_tracks.values():
                    if int(track.get("stable_id", -1)) == new_sid:
                        self._household_apply_identity_meta(track, sid=new_sid, kind="resident")
                self._recompute_sid_centroid(new_sid)
            actual_count = int(len(self.gallery.get(new_sid, [])))
            if int(rec.embedding_count) != actual_count:
                rec = self._resident_registry.patch(
                    str(rec.uuid),
                    embedding_count=actual_count,
                )
            self._promote_resident_count += 1
            self._resident_registry.save()
            self.save_gallery()
            return rec.to_dict()

    def patch_resident(
        self,
        resident_uuid: str,
        *,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._resident_registry is None:
                raise RuntimeError("household mode not enabled")
            rec = self._resident_registry.patch(str(resident_uuid), display_name=display_name)
            sid = int(rec.stable_id)
            for track in self.active_tracks.values():
                if int(track.get("stable_id", -1)) == sid:
                    self._household_apply_identity_meta(track, sid=sid, kind="resident")
            return rec.to_dict()

    def delete_resident(self, resident_uuid: str) -> Dict[str, Any]:
        with self._lock:
            if self._resident_registry is None:
                raise RuntimeError("household mode not enabled")
            rec = self._resident_registry.delete(str(resident_uuid))
            old_sid = int(rec.stable_id)
            now = float(time.time())
            has_live_ref = False
            for track in self.active_tracks.values():
                try:
                    if int(track.get("stable_id", -1)) == old_sid:
                        has_live_ref = True
                        break
                except Exception:
                    continue
            if not has_live_ref:
                for dq in self.ghosts.values():
                    for ghost in dq:
                        try:
                            if int(ghost.get("stable_id", -1)) == old_sid:
                                has_live_ref = True
                                break
                        except Exception:
                            continue
                    if has_live_ref:
                        break
            if has_live_ref:
                new_sid = int(self._household_mint_visitor(now))
                self._remap_sid(
                    old_sid,
                    new_sid,
                    kind="visitor",
                    purge_old=True,
                    # Removing an enrollment deletes its biometric gallery.
                    # The live visitor may accumulate fresh visitor exemplars
                    # later, but resident anchors are not copied across delete.
                    move_gallery=False,
                )
            else:
                self.gallery.pop(old_sid, None)
                self.sid_centroid.pop(old_sid, None)
                self.pose_gallery.pop(old_sid, None)
                self.pose_centroid.pop(old_sid, None)
                self.pose_last_seen.pop(old_sid, None)
                self.active_zones.pop(old_sid, None)
            self.save_gallery()
            return rec.to_dict()

    def get_identity_health(self) -> Dict[str, Any]:
        metrics = self.get_sid_metrics()
        with self._lock:
            residents = self.list_residents() if self._resident_registry is not None else []
        return {
            "household_mode": bool(metrics.get("household_mode")),
            "resident_count": int(metrics.get("resident_count", 0) or 0),
            "visitor_count": int(metrics.get("visitor_count", 0) or 0),
            "provisional_count": int(metrics.get("provisional_count", 0) or 0),
            "provisional_event_count": int(metrics.get("provisional_event_count", 0) or 0),
            "provisional_active_count": int(metrics.get("provisional_active_count", 0) or 0),
            "mint_visitor_count": int(metrics.get("mint_visitor_count", 0) or 0),
            "promote_resident_count": int(metrics.get("promote_resident_count", 0) or 0),
            "false_share_blocked_count": int(metrics.get("false_share_blocked_count", 0) or 0),
            "overlap_permit_grant_count": int(metrics.get("overlap_permit_grant_count", 0) or 0),
            "overlap_permit_deny_count": int(metrics.get("overlap_permit_deny_count", 0) or 0),
            "gallery_ids": int(metrics.get("gallery_ids", 0) or 0),
            "gallery_quality_reject_count": int(metrics.get("gallery_quality_reject_count", 0) or 0),
            "mnn_reject_count": int(metrics.get("mnn_reject_count", 0) or 0),
            "active_unique": int(metrics.get("active_unique", 0) or 0),
            "residents": residents,
            "metrics": metrics,
        }

    # --------------- Allocator -----------------
    def _sid_pool_soft_cap(self) -> int:
        base = max(1, int(self.max_total_ids))
        return int(max(16, base * 4))

    def _sid_owned_or_reserved(self, sid: int) -> bool:
        """Return whether the generic allocator must not hand out ``sid``."""
        sid_int = int(sid)
        if sid_int <= 0:
            return True
        if self._is_alias_reserved(sid_int) or sid_int in {int(v) for v in self.sid_alias.values()}:
            return True
        if self._resident_registry is not None and sid_int in self._household_resident_ids():
            return True
        if self._visitor_pool is not None and sid_int in self._visitor_pool.used_sids():
            return True
        if self._provisional_pool is not None and sid_int in self._provisional_pool.used_sids():
            return True
        if sid_int in self.gallery or sid_int in self.pose_gallery or sid_int in self.active_zones:
            return True
        for rec in self.active_tracks.values():
            try:
                if int(rec.get("stable_id", -1)) == sid_int:
                    return True
            except Exception:
                continue
        return False

    def _alloc_sid(self) -> int:
        # Keep allocator compact by preferring the lowest valid SID across
        # monotonic-next and free-pool candidates.
        while self._free_sids:
            sid_top = int(self._free_sids[0])
            if self._sid_owned_or_reserved(int(sid_top)):
                heapq.heappop(self._free_sids)
                self._free_sids_set.discard(int(sid_top))
                continue
            break

        next_sid = int(self.next_stable_id)
        while self._sid_owned_or_reserved(int(next_sid)):
            next_sid += 1

        free_sid: Optional[int] = int(self._free_sids[0]) if self._free_sids else None
        if free_sid is None or int(next_sid) < int(free_sid):
            self.next_stable_id = int(next_sid) + 1
            self.sid_global_first_seen.pop(int(next_sid), None)
            return int(next_sid)

        sid = int(heapq.heappop(self._free_sids))
        self._free_sids_set.discard(int(sid))
        if int(sid) >= int(self.next_stable_id):
            self.next_stable_id = int(sid) + 1
        self.sid_global_first_seen.pop(int(sid), None)
        return int(sid)

    def _free_sid(self, sid: int) -> None:
        try:
            sid = int(sid)
        except Exception:
            return
        if sid <= 0:
            return
        # Household ID spaces have dedicated lifecycle managers. Feeding them
        # into the legacy free heap enables resident/visitor/provisional reuse.
        if self.household_mode:
            return
        if self._is_alias_reserved(int(sid)):
            return
        if sid in self._free_sids_set:
            return
        heapq.heappush(self._free_sids, sid)
        self._free_sids_set.add(sid)
        self._save_sid_pool()

    def _try_recycle_unclaimed_sid(self, sid: Optional[int], *, keep_sid: Optional[int] = None) -> None:
        try:
            sid_int = int(sid) if sid is not None else 0
        except Exception:
            sid_int = 0
        if sid_int <= 0:
            return
        try:
            keep_int = int(keep_sid) if keep_sid is not None else None
        except Exception:
            keep_int = None
        if keep_int is not None and sid_int == keep_int:
            return
        try:
            if any(int(rec.get("stable_id", -1)) == sid_int for rec in self.active_tracks.values()):
                return
        except Exception:
            return
        try:
            if len(self.gallery.get(int(sid_int), [])) > 0:
                return
        except Exception:
            return
        try:
            if len(self.pose_gallery.get(int(sid_int), [])) > 0:
                return
        except Exception:
            return
        try:
            for dq in self.ghosts.values():
                for ghost in dq:
                    try:
                        if int(ghost.get("stable_id", -1)) == sid_int:
                            return
                    except Exception:
                        continue
        except Exception:
            return
        self._purge_sid_state(int(sid_int))
        free_before = int(len(self._free_sids_set))
        self._free_sid(int(sid_int))
        free_after = int(len(self._free_sids_set))
        if free_after > free_before:
            self._sid_pending_recycled_count += 1

    def _clear_pending_new_state(self, key: Tuple[int, int], *, keep_sid: Optional[int] = None) -> None:
        pending_sid = self._pending_new_sids.pop(key, None)
        self._pending_new_counts.pop(key, None)
        self._pending_new_ts.pop(key, None)
        self._pending_id_diag.pop(key, None)
        self._household_emb_counts.pop(key, None)
        if self.household_mode and pending_sid is not None:
            try:
                pending_int = int(pending_sid)
            except Exception:
                pending_int = 0
            if pending_int > 0 and self._household_is_provisional_sid(pending_int):
                if keep_sid is None or int(keep_sid) != pending_int:
                    self._household_release_provisional(pending_int)
        self._try_recycle_unclaimed_sid(pending_sid, keep_sid=keep_sid)

    def _purge_sid_state(self, sid: int) -> None:
        try:
            sid = int(sid)
        except Exception:
            return
        try:
            self.gallery.pop(sid, None)
            self.sid_centroid.pop(sid, None)
            self.sid_last_bbox.pop(sid, None)
            self.sid_last_bbox_sensor.pop(sid, None)
            self.sid_last_brightness.pop(sid, None)
            self.sid_last_color.pop(sid, None)
            self.active_zones.pop(sid, None)
            self.pose_gallery.pop(sid, None)
            self.pose_centroid.pop(sid, None)
            self.pose_last_seen.pop(sid, None)
            self.sid_global_last_seen.pop(sid, None)
            self.sid_global_first_seen.pop(sid, None)
            # Remove any dangling active track records tied to this SID.
            for key, rec in list(self.active_tracks.items()):
                try:
                    rec_sid = int(rec.get("stable_id", -1))
                except Exception:
                    rec_sid = -1
                if rec_sid == sid:
                    self.active_tracks.pop(key, None)
                    self._pending_id_diag.pop(key, None)
            # Drop ghost entries that still reference this SID.
            for sensor_id, dq in list(self.ghosts.items()):
                if not dq:
                    continue
                filtered = []
                for ghost in dq:
                    try:
                        g_sid = int(ghost.get("stable_id", -1))
                    except Exception:
                        g_sid = -1
                    if g_sid != sid:
                        filtered.append(ghost)
                if len(filtered) != len(dq):
                    self.ghosts[int(sensor_id)] = deque(filtered, maxlen=dq.maxlen)
            # Prune copresence pairs involving this SID.
            for pair in list(self.sid_last_copresent.keys()):
                try:
                    if int(pair[0]) == sid or int(pair[1]) == sid:
                        self.sid_last_copresent.pop(pair, None)
                except Exception:
                    self.sid_last_copresent.pop(pair, None)
        except Exception:
            pass

    def _load_sid_pool(self) -> None:
        try:
            if not os.path.exists(self.sid_pool_file):
                return
            if self.household_mode:
                secure_private_file(self.sid_pool_file)
            with open(self.sid_pool_file, "rb") as f:
                data = strict_json_loads(f.read(), label="stable-ID free SID pool")
            pool = data.get('free_sids', []) if isinstance(data, dict) else []
            pool = [int(x) for x in pool if isinstance(x, int) and x > 0]
            soft_cap = int(self._sid_pool_soft_cap())
            pool = [int(sid) for sid in pool if int(sid) <= soft_cap]
            pool = sorted(set(pool))[:32]
            for sid in pool:
                if sid not in self._free_sids_set:
                    heapq.heappush(self._free_sids, sid)
                    self._free_sids_set.add(sid)
        except Exception:
            pass

    def _clear_sid_pool_file(self) -> None:
        try:
            if os.path.exists(self.sid_pool_file):
                os.remove(self.sid_pool_file)
        except Exception:
            pass

    def _save_sid_pool(self) -> None:
        try:
            soft_cap = int(self._sid_pool_soft_cap())
            pool = sorted(int(sid) for sid in self._free_sids_set if int(sid) <= soft_cap)[:32]
            if self.household_mode:
                atomic_write_json_private(self.sid_pool_file, {'free_sids': pool})
            else:
                parent = os.path.dirname(self.sid_pool_file)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.sid_pool_file, 'w') as f:
                    json.dump({'free_sids': pool}, f)
        except Exception:
            pass

    # --------------- Long-term gallery persistence -----------------
    def _load_gallery(self) -> None:
        """Restore persisted identity gallery (long-term ReID memory)."""
        path = self.gallery_persist_file
        if not path or not os.path.exists(path):
            return
        try:
            if self.household_mode:
                secure_private_file(path)
            with np.load(path, allow_pickle=False) as data:
                sids = [int(s) for s in np.asarray(data.get("sids", []), dtype=np.int64).reshape(-1)]
                now = time.time()
                loaded = 0
                for sid in sids:
                    if sid <= 0:
                        continue
                    emb_key = f"emb_{sid}"
                    ts_key = f"ts_{sid}"
                    if emb_key not in data or ts_key not in data:
                        continue
                    embs = np.asarray(data[emb_key], dtype=np.float32)
                    tss = np.asarray(data[ts_key], dtype=np.float64).reshape(-1)
                    if embs.ndim != 2 or embs.shape[0] != tss.shape[0] or embs.shape[0] < 1:
                        continue
                    last_seen = float(np.max(tss))
                    if self.gallery_persist_max_age_s > 0.0 and (now - last_seen) > self.gallery_persist_max_age_s:
                        continue
                    dq = self.gallery[sid]
                    for i in range(embs.shape[0]):
                        vec = embs[i].reshape(-1)
                        n = float(np.linalg.norm(vec) + 1e-12)
                        dq.append((float(tss[i]), (vec / n).astype(np.float32)))
                    cen_key = f"centroid_{sid}"
                    if cen_key in data:
                        cen = np.asarray(data[cen_key], dtype=np.float32).reshape(-1)
                        n = float(np.linalg.norm(cen) + 1e-12)
                        self.sid_centroid[sid] = (cen / n).astype(np.float32)
                    else:
                        self._recompute_sid_centroid(sid)
                    fs_key = f"first_seen_{sid}"
                    if fs_key in data:
                        self.sid_global_first_seen[sid] = float(np.asarray(data[fs_key]).reshape(-1)[0])
                    self.sid_global_last_seen[sid] = last_seen
                    # Loaded SIDs must never be handed out by the allocator.
                    if sid in self._free_sids_set:
                        self._free_sids_set.discard(sid)
                        self._free_sids = [s for s in self._free_sids if int(s) != sid]
                        heapq.heapify(self._free_sids)
                    if sid >= self.next_stable_id:
                        self.next_stable_id = sid + 1
                    loaded += 1
                self._gallery_loaded_sids = int(loaded)
        except Exception:
            logger.warning("StableID gallery load failed (%s); starting with empty memory", path, exc_info=True)

    def save_gallery(self, *, save_registry: bool = True) -> bool:
        """Persist the identity gallery for long-term ReID memory.

        Gallery compression and replacement are performed outside ``_lock``.
        ``save_registry`` is disabled for periodic autosaves so their worker
        never performs resident-registry I/O on behalf of the media path;
        explicit and shutdown saves retain the registry update behavior.
        """
        path = self.gallery_persist_file
        if not path:
            return False
        try:
            registry_changed = False
            with self._lock:
                now = time.time()
                arrays: Dict[str, np.ndarray] = {}
                sids_out: List[int] = []
                for sid in list(self.gallery.keys()):
                    sid_int = int(sid)
                    if sid_int <= 0:
                        continue
                    if self.aliases_enabled and self.canonical_sid(sid_int) != sid_int:
                        continue
                    entries = [(float(t), e) for (t, e) in self.gallery.get(sid_int, []) if e is not None]
                    if not entries:
                        continue
                    last_seen = float(self.sid_global_last_seen.get(sid_int, entries[-1][0]))
                    if self.gallery_persist_max_age_s > 0.0 and (now - last_seen) > self.gallery_persist_max_age_s:
                        continue
                    arrays[f"emb_{sid_int}"] = np.stack([e for (_t, e) in entries], axis=0).astype(np.float32)
                    arrays[f"ts_{sid_int}"] = np.asarray([t for (t, _e) in entries], dtype=np.float64)
                    cen = self.sid_centroid.get(sid_int)
                    if cen is not None:
                        arrays[f"centroid_{sid_int}"] = np.asarray(cen, dtype=np.float32)
                    fs = self.sid_global_first_seen.get(sid_int)
                    if fs is not None:
                        arrays[f"first_seen_{sid_int}"] = np.asarray([float(fs)], dtype=np.float64)
                    sids_out.append(sid_int)
                arrays["sids"] = np.asarray(sids_out, dtype=np.int64)
                arrays["saved_ts"] = np.asarray([now], dtype=np.float64)
                if self.household_mode and self._resident_registry is not None:
                    registry_changed = False
                    for sid, resident in self._resident_registry.residents_by_sid.items():
                        actual_count = int(len(self.gallery.get(int(sid), [])))
                        if int(resident.embedding_count) != actual_count:
                            resident.embedding_count = actual_count
                            registry_changed = True
                    # The registry write is intentionally deferred until after
                    # the manager lock is released below. Autosave workers pass
                    # save_registry=False and leave this small metadata update
                    # in memory for the next explicit/shutdown save.
            with self._gallery_save_order_lock:
                self._gallery_save_generation += 1
                save_generation = int(self._gallery_save_generation)
            if registry_changed and save_registry and self._resident_registry is not None:
                self._resident_registry.save()
            dir_path = os.path.dirname(path)
            if dir_path:
                if self.household_mode:
                    ensure_private_directory(dir_path)
                else:
                    os.makedirs(dir_path, exist_ok=True)
            return self._write_gallery_snapshot(path, arrays, save_generation)
        except Exception:
            logger.warning("StableID gallery save failed (%s)", path, exc_info=True)
            return False

    def _write_gallery_snapshot(
        self,
        path: str,
        arrays: Mapping[str, np.ndarray],
        save_generation: int,
    ) -> bool:
        """Atomically write one ordered snapshot without taking ``_lock``."""
        # Explicit saves and background autosaves share one temp path; the
        # generation check prevents an older snapshot from overwriting a newer
        # one if filesystem-lock acquisition happens out of order.
        with self._gallery_file_write_lock:
            with self._gallery_save_order_lock:
                if int(save_generation) < int(self._gallery_last_written_generation):
                    return True
            tmp_path = f"{path}.tmp"
            if self.household_mode:
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PRIVATE_FILE_MODE)
                with os.fdopen(fd, "wb") as f:
                    np.savez_compressed(f, **arrays)
                os.chmod(tmp_path, PRIVATE_FILE_MODE)
            else:
                with open(tmp_path, "wb") as f:
                    np.savez_compressed(f, **arrays)
            os.replace(tmp_path, path)
            if self.household_mode:
                os.chmod(path, PRIVATE_FILE_MODE)
            with self._gallery_save_order_lock:
                self._gallery_last_written_generation = max(
                    int(self._gallery_last_written_generation),
                    int(save_generation),
                )
        return True

    def _autosave_gallery_worker(self) -> None:
        try:
            self.save_gallery(save_registry=False)
        finally:
            with self._gallery_autosave_lock:
                self._gallery_autosave_thread = None

    def _maybe_autosave_gallery(self, now_ts: float) -> None:
        if not self.gallery_persist_file or self.gallery_autosave_interval_s <= 0.0:
            return
        if (float(now_ts) - float(self._gallery_last_save_ts)) < self.gallery_autosave_interval_s:
            return
        self._gallery_last_save_ts = float(now_ts)
        # A single in-flight writer is enough: later due points do not create
        # unbounded persistence work or block the media callback.
        with self._gallery_autosave_lock:
            thread = self._gallery_autosave_thread
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(
                target=self._autosave_gallery_worker,
                name="StableIDGalleryAutosave",
                daemon=True,
            )
            self._gallery_autosave_thread = thread
            thread.start()

    def _gallery_add(self, sid: int, ts: float, emb: np.ndarray) -> None:
        """Add an embedding to a SID's gallery, preserving exemplar diversity."""
        dq = self.gallery[int(sid)]
        if self.household_mode:
            add_clustered_exemplar(dq, float(ts), emb)
            return
        if dq.maxlen is None or len(dq) < dq.maxlen:
            dq.append((float(ts), emb))
            return
        try:
            entries = list(dq)
            mat = np.stack([e for (_t, e) in entries], axis=0).astype(np.float32)
            sims = mat @ np.asarray(emb, dtype=np.float32).reshape(-1)
            idx = int(np.argmax(sims))
            entries[idx] = (float(ts), emb)
            dq.clear()
            dq.extend(entries)
        except Exception:
            dq.append((float(ts), emb))

    def _maybe_gallery_add(
        self,
        sid: int,
        ts: float,
        emb: np.ndarray,
        *,
        bbox: Optional[BBox] = None,
        blur_var: Optional[float] = None,
        pose_quality: Optional[Dict[str, float]] = None,
        identity_state: Optional[str] = None,
        identity_quality: Optional[str] = None,
    ) -> bool:
        sid_int = int(sid)
        accept, _reason = should_accept_gallery_embedding(
            household_mode=bool(self.household_mode),
            bbox=bbox,
            blur_var=blur_var,
            pose_quality=pose_quality,
            identity_state=identity_state,
            identity_quality=identity_quality,
            sid_has_gallery=bool(len(self.gallery.get(sid_int, []))),
            min_crop_h=int(self.min_crop_h),
            min_laplacian_var=float(self.min_laplacian_var),
        )
        if not accept:
            self._gallery_quality_reject_count += 1
            return False
        self._gallery_add(sid_int, float(ts), emb)
        return True

    def _persist_track_embedding(
        self,
        sid: int,
        ts: float,
        emb: np.ndarray,
        *,
        rec: Optional[Dict[str, Any]],
        bbox: BBox,
        pose_quality: Optional[Dict[str, float]] = None,
        blur_var: Optional[float] = None,
    ) -> bool:
        if self.household_mode and rec is not None and not self._household_should_persist_gallery(rec):
            return False
        identity_state: Optional[str] = None
        identity_quality: Optional[str] = None
        if rec is not None:
            if rec.get("identity_state"):
                identity_state = str(rec.get("identity_state"))
            if rec.get("identity_quality"):
                identity_quality = str(rec.get("identity_quality"))
            if self.household_mode and not identity_state:
                identity_state = self._household_identity_state(rec)
        return self._maybe_gallery_add(
            int(sid),
            float(ts),
            emb,
            bbox=bbox,
            blur_var=blur_var,
            pose_quality=pose_quality,
            identity_state=identity_state,
            identity_quality=identity_quality,
        )

    def _topology_handoff_allows_relax(
        self,
        sid: int,
        sensor_id: Optional[int],
    ) -> bool:
        if sensor_id is None:
            return True
        last_sensor = self.sid_last_bbox_sensor.get(int(sid))
        if last_sensor is None:
            return True
        if int(last_sensor) == int(sensor_id):
            return True
        return bool(self._camera_topology.is_overlap_pair(int(last_sensor), int(sensor_id)))

    def _gallery_match_candidates(self) -> List[int]:
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
        return candidates

    def _reject_household_gallery_match(
        self,
        *,
        track_key: Tuple[int, int],
        emb: np.ndarray,
        best_sid: int,
        ts: float,
        sensor_id: int,
        world_xy: Optional[Tuple[float, float]],
        world_valid: bool,
        appearance_sim: float,
    ) -> Tuple[bool, Optional[str]]:
        if not self.household_mode:
            return False, None

        self._frame_assignment.reset_if_new_frame(float(ts))
        if self._frame_assignment.is_claimed(int(best_sid), exclude_key=track_key):
            other_key = self._frame_assignment.claimed_by(int(best_sid))
            if other_key is not None and int(other_key[0]) != int(sensor_id):
                dual_ok = self._allow_cross_camera_sid_active(
                    int(best_sid),
                    int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    ts=float(ts),
                    appearance_sim=float(appearance_sim),
                )
                if not dual_ok:
                    self._mnn_reject_count += 1
                    return True, "frame_sid_claimed"

        peers = active_embeddings_same_frame(
            self.active_tracks,
            float(ts),
            epsilon_s=float(self.same_frame_sid_epsilon_s),
        )
        if track_key not in peers:
            peers[track_key] = np.asarray(emb, dtype=np.float32).reshape(-1)
        if len(peers) > 1:
            candidates = self._gallery_match_candidates()
            if not is_mutual_nearest_match(
                track_key,
                emb,
                int(best_sid),
                peers,
                candidates,
                self._similarity_for_candidates,
            ):
                self._mnn_reject_count += 1
                return True, "mnn_conflict"
        return False, None

    def _register_gallery_match_claim(
        self,
        *,
        track_key: Tuple[int, int],
        sid: int,
        ts: float,
    ) -> None:
        if not self.household_mode:
            return
        self._frame_assignment.reset_if_new_frame(float(ts))
        self._frame_assignment.claim(int(sid), track_key, float(ts))

    def _gallery_match_ok(
        self,
        *,
        track_key: Tuple[int, int],
        emb: Optional[np.ndarray],
        g_id: int,
        g_reid: float,
        g_req: float,
        sensor_id: int,
        bbox: BBox,
        ts: float,
        world_xy: Optional[Tuple[float, float]],
        world_valid: bool,
        exclude_key: Optional[Tuple[int, int]] = None,
        require_mnn_emb: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        can_take = self._allow_active_sid_match(
            sensor_id=int(sensor_id),
            sid=int(g_id),
            score=float(g_reid),
            required=float(g_req),
            bbox=bbox,
            ts=float(ts),
            exclude_key=exclude_key,
        )
        if not can_take:
            self._sid_guard_reject_count += 1
            if self._sid_claimed_same_frame(
                sensor_id=int(sensor_id),
                sid=int(g_id),
                ts=float(ts),
                exclude_key=exclude_key,
            ):
                return False, "same_frame_sid_conflict"
            return False, "active_guard"
        dual_ok = self._allow_cross_camera_sid_active(
            int(g_id),
            int(sensor_id),
            world_xy=world_xy,
            world_valid=bool(world_valid),
            ts=float(ts),
            appearance_sim=float(g_reid),
        )
        if not dual_ok:
            return False, "exclusivity_blocked"
        if self.household_mode:
            if require_mnn_emb and emb is not None:
                reject, reason = self._reject_household_gallery_match(
                    track_key=track_key,
                    emb=emb,
                    best_sid=int(g_id),
                    ts=float(ts),
                    sensor_id=int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    appearance_sim=float(g_reid),
                )
                if reject:
                    return False, reason
            else:
                # Pose-only / no-emb path: still enforce frame SID claims.
                self._frame_assignment.reset_if_new_frame(float(ts))
                if self._frame_assignment.is_claimed(int(g_id), exclude_key=track_key):
                    other_key = self._frame_assignment.claimed_by(int(g_id))
                    if other_key is not None and int(other_key[0]) != int(sensor_id):
                        dual_claim_ok = self._allow_cross_camera_sid_active(
                            int(g_id),
                            int(sensor_id),
                            world_xy=world_xy,
                            world_valid=bool(world_valid),
                            ts=float(ts),
                            appearance_sim=float(g_reid),
                        )
                        if not dual_claim_ok:
                            self._mnn_reject_count += 1
                            return False, "frame_sid_claimed"
            self._register_gallery_match_claim(track_key=track_key, sid=int(g_id), ts=float(ts))
        return True, None

    def _load_aliases(self) -> None:
        if not self.aliases_enabled:
            return
        try:
            if not self.alias_file or not os.path.exists(self.alias_file):
                return
            if self.household_mode:
                secure_private_file(self.alias_file)
            with open(self.alias_file, "rb") as f:
                data = strict_json_loads(f.read(), label="stable-ID alias registry")
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
            if self.household_mode:
                atomic_write_json_private(self.alias_file, payload)
            else:
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
        # Reserve only alias sources; canonical destination IDs stay allocatable.
        self.sid_alias_reserved = set(self.sid_alias.keys())

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
            # Scale/brightness/spatial penalties are same-camera signals only;
            # applying them across cameras (different geometry/lighting) was
            # suppressing legitimate cross-camera handoffs.
            same_sensor_ref = (
                sensor_id is not None
                and self.sid_last_bbox_sensor.get(int(sid)) is not None
                and int(self.sid_last_bbox_sensor[int(sid)]) == int(sensor_id)
            )
            if self.adaptive_penalty and curr_bbox is not None and same_sensor_ref:
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
            if self.spatial_penalty and sensor_id is not None and curr_bbox is not None and same_sensor_ref:
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
                        if (not self.household_mode) or self._topology_handoff_allows_relax(
                            int(sid), sensor_id
                        ):
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

    def _record_id_event(self, event: str) -> None:
        try:
            key = str(event).strip()
        except Exception:
            return
        if not key:
            return
        self._id_event_counts[key] = int(self._id_event_counts.get(key, 0)) + 1

    def _set_track_diag(
        self,
        key: Tuple[int, int],
        *,
        event: str,
        reject_reason: Optional[str],
        embedding_present: bool,
        pose_present: bool,
        sid_candidate: Optional[int],
        stable_id: Optional[int],
        ts: float,
        reid_confidence: Optional[float] = None,
        reid_required: Optional[float] = None,
        overlap_permit: Optional[bool] = None,
        identity_kind: Optional[str] = None,
        identity_state: Optional[str] = None,
        resident_uuid: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "id_event": str(event),
            "id_reject_reason": str(reject_reason) if reject_reason else None,
            "embedding_present": bool(embedding_present),
            "pose_present": bool(pose_present),
            "sid_candidate": int(sid_candidate) if sid_candidate is not None else None,
            "stable_id": int(stable_id) if stable_id is not None else None,
            "ts": float(ts),
            "reid_confidence": float(reid_confidence) if reid_confidence is not None else None,
            "reid_required": float(reid_required) if reid_required is not None else None,
            "overlap_permit": bool(overlap_permit) if overlap_permit is not None else None,
        }
        if self.household_mode and stable_id is not None:
            kind = identity_kind or self._household_identity_kind(int(stable_id))
            diag["identity_kind"] = str(kind)
            diag["identity_state"] = str(identity_state or identity_state_for_kind(kind))
            if resident_uuid is not None:
                diag["resident_uuid"] = resident_uuid
            elif self._resident_registry is not None and kind == "resident":
                diag["resident_uuid"] = self._resident_registry.uuid_for(int(stable_id))
            else:
                diag["resident_uuid"] = None
            if display_name is not None:
                diag["display_name"] = display_name
            elif self._resident_registry is not None and kind == "resident":
                diag["display_name"] = self._resident_registry.display_name_for(int(stable_id))
            else:
                diag["display_name"] = None
            if self._visitor_pool is not None and kind == "visitor":
                diag["visitor_generation"] = self._visitor_pool.generation_for(int(stable_id))
        self._pending_id_diag[key] = dict(diag)
        return diag

    def _record_sid_sensor_world(
        self,
        sid: int,
        sensor_id: int,
        *,
        world_xy: Optional[Tuple[float, float]],
        world_valid: bool,
        ts: float,
    ) -> None:
        if world_xy is None:
            return
        try:
            wx = float(world_xy[0])
            wy = float(world_xy[1])
        except Exception:
            return
        self._sid_sensor_world[(int(sid), int(sensor_id))] = {
            "world_xy": (wx, wy),
            "world_valid": bool(world_valid),
            "ts": float(ts),
        }

    def _other_active_sensors_for_sid(self, sid: int, sensor_id: int) -> List[int]:
        pairs = self.active_zones.get(int(sid), set())
        out: List[int] = []
        for sid_sensor, _zone in pairs:
            if int(sid_sensor) != int(sensor_id):
                out.append(int(sid_sensor))
        return out

    def validate_camera_topology(
        self,
        camera_labels: Mapping[int, str],
        *,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Validate configured camera IDs against the runtime's active map."""
        with self._lock:
            errors = self._camera_topology.validate_runtime_map(camera_labels)
            if self._camera_topology.source_path is None:
                errors.append("camera topology file is not loaded")
            errors = sorted(set(str(error) for error in errors))
            self._camera_topology_validation_errors = errors
            self._camera_topology_validation_status = "invalid" if errors else "valid"
            result = {
                "valid": not bool(errors),
                "status": self._camera_topology_validation_status,
                "errors": list(errors),
                "camera_topology_file": self._camera_topology.source_path,
            }
            if strict and errors:
                raise ValueError("camera topology/runtime map mismatch: " + "; ".join(errors))
            return result

    def overlap_permit(
        self,
        sid: int,
        sensor_id: int,
        world_xy: Optional[Tuple[float, float]],
        ts: float,
        *,
        world_valid: bool = False,
        appearance_sim: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Return whether sid may be active on sensor_id while active elsewhere.

        Implements the algorithm in plans/household_identity/camera_topology.md.
        """
        sid_int = int(sid)
        sensor_int = int(sensor_id)
        other_sensors = self._other_active_sensors_for_sid(sid_int, sensor_int)
        if not other_sensors:
            return True, None

        try:
            ts_f = float(ts)
        except Exception:
            ts_f = 0.0

        curr_world: Optional[Tuple[float, float]] = None
        if world_xy is not None:
            try:
                curr_world = (float(world_xy[0]), float(world_xy[1]))
            except Exception:
                curr_world = None

        for other_sensor in other_sensors:
            params = self._camera_topology.overlap_params(other_sensor, sensor_int)
            if params is None:
                return False, "no_overlap_pair"

            other_state = self._sid_sensor_world.get((sid_int, int(other_sensor)), {})
            other_world = other_state.get("world_xy")
            other_valid = bool(other_state.get("world_valid", False))
            other_ts = float(other_state.get("ts", ts_f))

            geometry_ok = (
                curr_world is not None
                and other_world is not None
                and bool(world_valid)
                and other_valid
            )

            if not geometry_ok:
                if not self.overlap_allow_appearance_only:
                    return False, "missing_world"
                if appearance_sim is None or float(appearance_sim) < 0.85:
                    return False, "appearance_only_insufficient"
                return True, "appearance_only"

            if abs(ts_f - other_ts) > float(params.max_time_delta_s):
                return False, "time_delta"

            try:
                ox, oy = float(other_world[0]), float(other_world[1])
                dist = math.hypot(float(curr_world[0]) - ox, float(curr_world[1]) - oy)
            except Exception:
                return False, "world_parse"

            if dist > float(params.max_world_dist_m):
                return False, "world_dist"

            if appearance_sim is not None and float(appearance_sim) < float(params.require_appearance_sim):
                return False, "appearance_sim"

        return True, None

    def _allow_cross_camera_sid_active(
        self,
        sid: int,
        sensor_id: int,
        *,
        world_xy: Optional[Tuple[float, float]] = None,
        world_valid: bool = False,
        ts: float = 0.0,
        appearance_sim: Optional[float] = None,
    ) -> bool:
        """Enforce global exclusivity with optional FoV-overlap exception."""
        sid_int = int(sid)
        if not self._other_active_sensors_for_sid(sid_int, int(sensor_id)):
            return True
        if self.allow_multi_zone_active and not self.household_mode:
            return True

        granted, reason = self.overlap_permit(
            sid_int,
            int(sensor_id),
            world_xy,
            float(ts),
            world_valid=bool(world_valid),
            appearance_sim=appearance_sim,
        )
        if granted:
            if reason == "appearance_only":
                self._overlap_permit_degraded_count += 1
            self._overlap_permit_grant_count += 1
            return True

        self._false_share_blocked_count += 1
        self._overlap_permit_deny_count += 1
        return False

    def _active_bbox_for_sid_on_sensor(
        self,
        sensor_id: int,
        sid: int,
        *,
        exclude_key: Optional[Tuple[int, int]] = None,
    ) -> Optional[BBox]:
        best_bbox: Optional[BBox] = None
        best_ts = float("-inf")
        sid_int = int(sid)
        sensor_int = int(sensor_id)
        for rec_key, rec in self.active_tracks.items():
            try:
                if int(rec_key[0]) != sensor_int:
                    continue
                if exclude_key is not None and rec_key == exclude_key:
                    continue
                if int(rec.get("stable_id", -1)) != sid_int:
                    continue
                bbox = rec.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                last_ts = float(rec.get("last_seen_ts", 0.0))
                if last_ts >= best_ts:
                    best_ts = last_ts
                    best_bbox = (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
            except Exception:
                continue
        return best_bbox

    def _sid_claimed_same_frame(
        self,
        *,
        sensor_id: int,
        sid: int,
        ts: Optional[float],
        exclude_key: Optional[Tuple[int, int]] = None,
    ) -> bool:
        if ts is None:
            return False
        try:
            sensor_int = int(sensor_id)
            sid_int = int(sid)
            ts_f = float(ts)
        except Exception:
            return False
        if sensor_int < 0 or sid_int <= 0:
            return False
        eps = float(self.same_frame_sid_epsilon_s)
        for rec_key, rec in self.active_tracks.items():
            try:
                if int(rec_key[0]) != sensor_int:
                    continue
                if exclude_key is not None and rec_key == exclude_key:
                    continue
                if int(rec.get("stable_id", -1)) != sid_int:
                    continue
                last_ts = float(rec.get("last_seen_ts", -1e18))
            except Exception:
                continue
            if abs(float(last_ts) - ts_f) <= eps:
                return True
        return False

    def _allow_active_sid_match(
        self,
        *,
        sensor_id: int,
        sid: int,
        score: float,
        required: float,
        bbox: BBox,
        ts: Optional[float] = None,
        exclude_key: Optional[Tuple[int, int]] = None,
    ) -> bool:
        # Strict same-frame uniqueness: never allow two tracks in one camera
        # frame to claim the same SID.
        if self._sid_claimed_same_frame(
            sensor_id=int(sensor_id),
            sid=int(sid),
            ts=ts,
            exclude_key=exclude_key,
        ):
            self._sid_same_frame_conflict_count += 1
            return False
        if not self.active_id_guard_strict:
            return True
        active_pairs = self.active_zones.get(int(sid), set())
        active_here = any(int(sid_sensor) == int(sensor_id) for (sid_sensor, _zone) in active_pairs)
        if not active_here:
            return True
        score_f = float(score)
        req_f = float(required)
        if score_f >= (req_f + float(self.active_id_guard_margin)):
            return True
        if score_f < (req_f - float(self.active_id_guard_relaxed_reid_slack)):
            return False

        ref_bbox = self._active_bbox_for_sid_on_sensor(int(sensor_id), int(sid), exclude_key=exclude_key)
        if ref_bbox is None:
            return False
        try:
            cx = float(bbox[0]) + float(bbox[2]) / 2.0
            cy = float(bbox[1]) + float(bbox[3]) / 2.0
            rx = float(ref_bbox[0]) + float(ref_bbox[2]) / 2.0
            ry = float(ref_bbox[1]) + float(ref_bbox[3]) / 2.0
            dist = float(((cx - rx) ** 2 + (cy - ry) ** 2) ** 0.5)
            diag = max(
                1.0,
                float((float(bbox[2]) ** 2 + float(bbox[3]) ** 2) ** 0.5),
                float((float(ref_bbox[2]) ** 2 + float(ref_bbox[3]) ** 2) ** 0.5),
            )
            dist_ratio = float(dist / diag)
            area_a = max(1.0, float(bbox[2]) * float(bbox[3]))
            area_b = max(1.0, float(ref_bbox[2]) * float(ref_bbox[3]))
            scale_ratio = area_a / area_b if area_a >= area_b else area_b / area_a
        except Exception:
            return False
        if dist_ratio > float(self.active_id_guard_relaxed_dist_ratio):
            return False
        if scale_ratio > float(self.active_id_guard_relaxed_scale_ratio):
            return False
        return True

    # --------------- Public API --------------
    def get_track_diagnostics(self, sensor_id: int, ds_obj_id: int) -> Dict[str, Any]:
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            if rec is not None:
                diag = rec.get("id_diag")
                if isinstance(diag, dict):
                    result = dict(diag)
                    for name in (
                        "identity_kind",
                        "identity_state",
                        "resident_uuid",
                        "display_name",
                        "visitor_generation",
                    ):
                        if rec.get(name) is not None:
                            result[name] = rec.get(name)
                    return result
            diag_pending = self._pending_id_diag.get(key)
            if isinstance(diag_pending, dict):
                return dict(diag_pending)
        return {}

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
        ignore_both_active: bool = False,
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
                if (not ignore_both_active) and self.active_zones.get(root_a) and self.active_zones.get(root_b):
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
                    if int(src_root) in self.sid_last_bbox_sensor:
                        self.sid_last_bbox_sensor[int(dst_root)] = self.sid_last_bbox_sensor[int(src_root)]
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

    def _canonical_sid_set(self) -> set[int]:
        """Collect canonical SID set across gallery, active tracks, ghosts, and last-seen state."""
        sids: set[int] = set()
        for sid in self.gallery.keys():
            try:
                sid_can = self.canonical_sid(int(sid))
            except Exception:
                continue
            if sid_can > 0:
                sids.add(int(sid_can))

        for rec in self.active_tracks.values():
            try:
                sid_can = self.canonical_sid(int(rec.get("stable_id")))
            except Exception:
                continue
            if sid_can > 0:
                sids.add(int(sid_can))

        for sid in self.sid_global_last_seen.keys():
            try:
                sid_can = self.canonical_sid(int(sid))
            except Exception:
                continue
            if sid_can > 0:
                sids.add(int(sid_can))

        for dq in self.ghosts.values():
            for ghost in dq:
                try:
                    sid_can = self.canonical_sid(int(ghost.get("stable_id")))
                except Exception:
                    continue
                if sid_can > 0:
                    sids.add(int(sid_can))

        return sids

    def _count_canonical_sids(self) -> int:
        """Count active/stable IDs after applying alias canonicalization."""
        return int(len(self._canonical_sid_set()))

    def _pressure_relief_recycle(self, now_ts: float, max_ids: int) -> int:
        """Force ID-pool relief without risky merges by recycling inactive weak/stale identities."""
        now = float(now_ts)
        target = int(max_ids)
        if target <= 0:
            return 0
        current = self._canonical_sid_set()
        if len(current) <= target:
            return 0

        active_can: set[int] = set()
        for rec in self.active_tracks.values():
            try:
                sid_can = self.canonical_sid(int(rec.get("stable_id", -1)))
            except Exception:
                sid_can = -1
            if sid_can > 0:
                active_can.add(int(sid_can))

        ghost_latest: Dict[int, float] = {}
        for dq in self.ghosts.values():
            for ghost in dq:
                try:
                    sid_can = self.canonical_sid(int(ghost.get("stable_id", -1)))
                    ts = float(ghost.get("ts", 0.0) or 0.0)
                except Exception:
                    continue
                if sid_can <= 0:
                    continue
                prev = ghost_latest.get(int(sid_can))
                if prev is None or ts > float(prev):
                    ghost_latest[int(sid_can)] = float(ts)

        age_gate_weak = max(1.0, float(self.active_evict_grace_s) * 0.5)
        age_gate_base = max(2.0, float(self.active_evict_grace_s))
        ranked: List[Tuple[int, float, int, int]] = []
        for sid in sorted(current):
            sid_int = int(sid)
            if sid_int in active_can:
                continue
            if self._is_alias_reserved(sid_int):
                continue
            emb_count = int(len(self.gallery.get(sid_int, [])))
            last_seen = self.sid_global_last_seen.get(sid_int)
            if last_seen is None:
                last_seen = ghost_latest.get(sid_int, 0.0)
            age = max(0.0, now - float(last_seen or 0.0))

            bucket = 99
            if emb_count < 1 and age >= age_gate_weak:
                bucket = 0
            elif emb_count <= 1 and age >= age_gate_base:
                bucket = 1
            elif age >= age_gate_base:
                bucket = 2
            if bucket >= 99:
                continue
            ranked.append((int(bucket), -float(age), int(emb_count), sid_int))

        ranked.sort()
        recycled = 0
        for _bucket, _neg_age, _emb_count, sid in ranked:
            if len(current) <= target:
                break
            self._purge_sid_state(int(sid))
            self._free_sid(int(sid))
            current.discard(int(sid))
            recycled += 1
            self._record_id_event("pressure_recycle")
        return int(recycled)

    def _auto_merge_if_needed(self, now_ts: float) -> int:
        """Apply automatic alias merges when canonical ID count exceeds configured limits."""
        if self.household_mode:
            self._auto_merge_last_reason = "household_mode_disabled"
            return 0
        if not self.auto_merge_enabled or not self.aliases_enabled:
            self._auto_merge_last_reason = "disabled"
            return 0

        now = float(now_ts)
        max_ids = int(self.max_total_ids)
        if max_ids <= 0:
            self._auto_merge_last_reason = "invalid_max_ids"
            return 0

        current_ids = self._count_canonical_sids()
        if current_ids <= max_ids:
            self._auto_merge_last_reason = "under_cap"
            self._auto_merge_last_pressure_recycled = 0
            return 0

        interval = float(self.auto_merge_interval_s)
        if interval > 0.0 and self._last_auto_merge_ts is not None and (now - float(self._last_auto_merge_ts)) < interval:
            self._auto_merge_last_reason = "interval_guard"
            return 0

        self._last_auto_merge_ts = now
        self._auto_merge_runs += 1
        self._auto_merge_triggers += 1

        remaining = int(self.auto_merge_max_attempts)
        if remaining <= 0:
            self._auto_merge_last_reason = "max_attempts_zero"
            return 0

        applied = 0
        blocked_count = 0
        candidate_count = 0
        stall_reason = "none"
        pressure_recycled = 0
        support_gate = int(max(1, self.auto_merge_min_embeddings_for_suggest))
        suggestion_limit = max(1, min(remaining, 25))
        min_sim_try = float(self.auto_merge_min_sim)
        relax_attempts = 0
        hard_guard_blocked = False
        while remaining > 0:
            candidates = self.suggest_aliases(
                min_sim=min_sim_try,
                limit=suggestion_limit,
                require_inactive=self.auto_merge_respect_inactive,
                now_ts=now,
                min_embeddings_for_suggest_override=support_gate,
            )
            if not candidates:
                if relax_attempts < 2 and min_sim_try > 0.86:
                    min_sim_try = max(0.86, float(min_sim_try) - 0.03)
                    relax_attempts += 1
                    stall_reason = "no_candidates_relaxed"
                    continue
                stall_reason = "no_candidates"
                break
            candidate_count += len(candidates)
            applied_this_round = False
            for cand in candidates:
                if remaining <= 0:
                    break
                remaining -= 1
                if not isinstance(cand, dict):
                    continue
                src_sid = int(cand.get("a", 0) or 0)
                dst_sid = int(cand.get("b", 0) or 0)
                if src_sid <= 0 or dst_sid <= 0:
                    continue
                if src_sid == dst_sid:
                    continue

                ignore_both_active = False
                if bool(cand.get("blocked")) and not self.auto_merge_force_stuck:
                    reason = str(cand.get("block_reason") or "")
                    sim_val = float(cand.get("sim") or 0.0)
                    if (
                        reason == "both_active"
                        and self.auto_merge_allow_both_active
                        and sim_val >= float(self.auto_merge_both_active_min_sim)
                    ):
                        ignore_both_active = True
                    else:
                        blocked_count += 1
                        stall_reason = "blocked"
                        self._auto_merge_suppressed += 1
                        continue

                canonical_src = self.canonical_sid(src_sid)
                canonical_dst = self.canonical_sid(dst_sid)
                if canonical_src == canonical_dst:
                    continue

                res = self.set_alias(
                    canonical_src,
                    canonical_dst,
                    force=self.auto_merge_force_stuck,
                    ignore_both_active=ignore_both_active,
                    now_ts=now,
                )
                if res.get("applied"):
                    applied += 1
                    applied_this_round = True
                    self._auto_merge_applied += 1
                else:
                    self._auto_merge_suppressed += 1
                    if str(res.get("reason") or "") == "copresent_recently":
                        # The highest-ranked candidate was proven to be two
                        # distinct people. Continuing down an ambiguous list can
                        # merge a fresh fragment into either person and conceal
                        # the conflict, so stop this pressure-relief round.
                        blocked_count += 1
                        stall_reason = "copresent_recently"
                        hard_guard_blocked = True
                        break

                if self._count_canonical_sids() <= max_ids:
                    break

            if hard_guard_blocked:
                break
            if not applied_this_round:
                if blocked_count >= candidate_count and candidate_count > 0:
                    stall_reason = "all_blocked"
                elif stall_reason == "none":
                    stall_reason = "insufficient_similarity"
                break
            if self._count_canonical_sids() <= max_ids:
                break

        self._auto_merge_last_candidate_count = int(candidate_count)
        self._auto_merge_last_blocked_count = int(blocked_count)
        self._auto_merge_last_support_gate = int(support_gate)
        current_after = self._count_canonical_sids()
        if applied < 1 and current_after > max_ids and candidate_count < 1:
            pressure_recycled = int(self._pressure_relief_recycle(now, max_ids))
            self._auto_merge_last_pressure_recycled = int(pressure_recycled)
            self._auto_merge_pressure_recycled_total += int(pressure_recycled)
            if pressure_recycled > 0:
                stall_reason = "pressure_recycle"
            current_after = self._count_canonical_sids()
        else:
            self._auto_merge_last_pressure_recycled = 0

        self._auto_merge_last_reason = str(stall_reason or "none")
        applied_effective = int(applied + pressure_recycled)
        if applied_effective < 1 and current_after > max_ids:
            self._auto_merge_zero_apply_streak += 1
            warn_interval_s = 120.0
            warn_streak = 3
            last_warn = self._auto_merge_last_warn_ts
            streak = int(self._auto_merge_zero_apply_streak)
            milestone = streak in (3, 6, 12) or (streak % 15 == 0)
            if streak >= warn_streak and (milestone or last_warn is None or (now - float(last_warn)) >= warn_interval_s):
                logger.warning(
                    "StableID auto-merge stalled: canonical=%d max=%d candidates=%d blocked=%d support_gate=%d streak=%d reason=%s recycled=%d",
                    int(current_after),
                    int(max_ids),
                    int(candidate_count),
                    int(blocked_count),
                    int(support_gate),
                    int(streak),
                    str(self._auto_merge_last_reason),
                    int(self._auto_merge_last_pressure_recycled),
                )
                self._auto_merge_last_warn_ts = float(now)
        else:
            self._auto_merge_zero_apply_streak = 0

        return int(applied_effective)

    def _sid_pair_similarity(self, sid_a: int, sid_b: int, centroids: Dict[int, np.ndarray]) -> float:
        """Similarity between two identities for duplicate detection.

        Uses the max of centroid similarity and a robust top-k mean of
        cross-exemplar similarities. Centroids blur multiple viewpoints
        together, which suppresses legitimate same-person merges across
        cameras; exemplar pairs recover them while the top-k mean damps
        single-outlier false matches.
        """
        cen_sim = -1.0
        ca = centroids.get(int(sid_a))
        cb = centroids.get(int(sid_b))
        if ca is not None and cb is not None:
            cen_sim = float(self._cosine(ca, cb))
        try:
            vecs_a = [v for (_ts, v) in self.gallery.get(int(sid_a), []) if v is not None]
            vecs_b = [v for (_ts, v) in self.gallery.get(int(sid_b), []) if v is not None]
            if vecs_a and vecs_b:
                mat_a = np.stack(vecs_a, axis=0).astype(np.float32)
                mat_b = np.stack(vecs_b, axis=0).astype(np.float32)
                if mat_a.shape[1] == mat_b.shape[1]:
                    cross = (mat_a @ mat_b.T).reshape(-1)
                    k = min(3, cross.size)
                    topk = np.partition(cross, cross.size - k)[cross.size - k :]
                    cen_sim = max(cen_sim, float(np.mean(topk)))
        except Exception:
            pass
        return float(cen_sim)

    def suggest_aliases(
        self,
        *,
        min_sim: Optional[float] = None,
        limit: int = 20,
        require_inactive: bool = True,
        now_ts: Optional[float] = None,
        min_embeddings_for_suggest_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        now_ts = float(now_ts if now_ts is not None else time.time())
        min_sim = float(self.suggest_min_sim if min_sim is None else min_sim)
        support_gate = int(
            max(
                1,
                self.min_embeddings_for_suggest
                if min_embeddings_for_suggest_override is None
                else int(min_embeddings_for_suggest_override),
            )
        )
        with self._lock:
            # Build canonical SID set with enough support.
            candidates: List[int] = []
            seen: set[int] = set()
            support_counts: Dict[int, int] = {}
            for sid in list(self.gallery.keys()):
                sid_int = int(sid)
                sid_can = self.canonical_sid(sid_int)
                if sid_can in seen:
                    continue
                if self._is_alias_src(sid_int):
                    continue
                seen.add(sid_can)
                emb_count = int(len(self.gallery.get(int(sid_can), [])))
                support_counts[int(sid_can)] = int(emb_count)
                if emb_count < int(support_gate):
                    continue
                candidates.append(int(sid_can))
            self._alias_support_ids_ge_1 = int(sum(1 for c in support_counts.values() if c >= 1))
            self._alias_support_ids_ge_2 = int(sum(1 for c in support_counts.values() if c >= 2))
            self._alias_support_ids_ge_3 = int(sum(1 for c in support_counts.values() if c >= 3))

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
            if len(candidates) < 2:
                self._alias_candidate_pool_size = 0
                self._alias_candidate_sim_p50 = None
                self._alias_candidate_sim_p95 = None
                return []

            best: Dict[int, Tuple[int, float, float]] = {}
            for sid in candidates:
                sims: List[Tuple[float, int]] = []
                for other in candidates:
                    if other == sid:
                        continue
                    sim = self._sid_pair_similarity(int(sid), int(other), centroids)
                    sims.append((float(sim), int(other)))
                if not sims:
                    continue
                sims.sort(key=lambda item: item[0], reverse=True)
                best_sid = int(sims[0][1])
                best_sim = float(sims[0][0])
                second_sim = float(sims[1][0]) if len(sims) > 1 else -1.0
                best[int(sid)] = (best_sid, best_sim, second_sim)

            sorted_candidates = sorted(int(sid) for sid in candidates)
            pair_pool_size = 0
            pair_sims: List[float] = []
            results: List[Dict[str, Any]] = []
            for idx, sid in enumerate(sorted_candidates):
                for best_sid in sorted_candidates[idx + 1 :]:
                    sim = float(self._sid_pair_similarity(int(sid), int(best_sid), centroids))
                    if not math.isfinite(sim):
                        continue
                    pair_pool_size += 1
                    pair_sims.append(float(sim))

                    pose_sim = self._pose_similarity_between_sids(int(sid), int(best_sid))
                    a_best, a_sim, a_second = best.get(int(sid), (None, -1.0, -1.0))
                    b_best, b_sim, b_second = best.get(int(best_sid), (None, -1.0, -1.0))
                    mutual_nn = a_best == int(best_sid) and b_best == int(sid)
                    gap_a = max(0.0, float(a_sim) - float(a_second))
                    gap_b = max(0.0, float(b_sim) - float(b_second))
                    mnn_gap = min(gap_a, gap_b)
                    strong_mnn = bool(mutual_nn and mnn_gap >= float(self.suggest_mnn_margin))
                    high_reid = bool(sim >= float(min_sim) + 0.03)
                    near_reid = bool(sim >= max(0.0, float(min_sim) - 0.04))
                    pose_boost = bool(
                        pose_sim is not None
                        and float(pose_sim) >= float(self.suggest_pose_sim_high)
                        and near_reid
                    )
                    sim_ok = bool((strong_mnn and sim >= float(min_sim)) or high_reid or pose_boost)
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

                    if pose_sim is not None and sim >= 0.94 and float(pose_sim) < float(self.suggest_pose_sim_low):
                        blocked = True
                        block_reason = "pose_mismatch"

                    count_a = int(len(self.gallery.get(int(sid), [])))
                    count_b = int(len(self.gallery.get(int(best_sid), [])))
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

                    score = float(sim)
                    if pose_sim is not None:
                        score += 0.04 * max(0.0, min(1.0, float(pose_sim)) - float(self.suggest_pose_sim_low))
                    if strong_mnn:
                        score += 0.01
                    if high_reid:
                        score += 0.01

                    results.append(
                        {
                            "a": int(sid),
                            "b": int(best_sid),
                            "score": float(score),
                            "sim": float(sim),
                            "pose_sim": float(pose_sim) if pose_sim is not None else None,
                            "mnn": bool(strong_mnn),
                            "mnn_gap": float(mnn_gap),
                            "canonical": int(min(sid, best_sid)),
                            "preferred_canonical": int(preferred),
                            "a_embedding_count": int(count_a),
                            "b_embedding_count": int(count_b),
                            "blocked": bool(blocked),
                            "block_reason": block_reason,
                        }
                    )

            self._alias_candidate_pool_size = int(pair_pool_size)
            self._alias_candidate_sim_p50 = self._percentile(pair_sims, 50.0) if pair_sims else None
            self._alias_candidate_sim_p95 = self._percentile(pair_sims, 95.0) if pair_sims else None

            results.sort(key=lambda item: float(item.get("score", item.get("sim", 0.0))), reverse=True)
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
        world_xy: Optional[Tuple[float, float]] = None,
        world_valid: bool = False,
    ) -> int:
        """Update or create stable_id for a DS track.

        If new, attempt to match ghost for same camera, else gallery (cross-camera).
        Optionally extracts embedding when due (new track or embed interval elapsed).
        """
        key = (int(sensor_id), int(ds_obj_id))
        with self._lock:
            rec = self.active_tracks.get(key)
            is_new = rec is None
            diag_event = "new_alloc" if is_new else "reuse_active"
            diag_reject_reason: Optional[str] = None
            diag_sid_candidate: Optional[int] = None
            diag_reid_confidence: Optional[float] = None
            diag_reid_required: Optional[float] = None
            diag_overlap_permit: Optional[bool] = None
            if is_new:
                self._auto_merge_if_needed(float(ts))
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

            # Hard uniqueness repair: if this existing track would create a
            # same-frame SID duplicate on this sensor, split it to a fresh SID.
            if rec is not None:
                try:
                    current_sid = int(rec.get("stable_id", -1))
                except Exception:
                    current_sid = -1
                if current_sid > 0 and self._sid_claimed_same_frame(
                    sensor_id=int(sensor_id),
                    sid=int(current_sid),
                    ts=float(ts),
                    exclude_key=key,
                ):
                    self._sid_same_frame_conflict_count += 1
                    old_sid = int(current_sid)
                    zone_existing = rec.get("zone", "default")
                    try:
                        pairs = self.active_zones.get(int(old_sid), set())
                        if (int(sensor_id), zone_existing) in pairs:
                            pairs.remove((int(sensor_id), zone_existing))
                        if pairs:
                            self.active_zones[int(old_sid)] = pairs
                        else:
                            self.active_zones.pop(int(old_sid), None)
                    except Exception:
                        pass
                    if self.household_mode:
                        new_sid = int(self._household_alloc_provisional(exclude_key=key))
                        self._household_apply_identity_meta(rec, sid=int(new_sid), kind="provisional")
                    else:
                        new_sid = int(self._alloc_sid())
                    self._sid_new_alloc_count += 1
                    self._sid_fragmentation_events += 1
                    if self.aliases_enabled:
                        new_sid = int(self.canonical_sid(int(new_sid)))
                    rec["stable_id"] = int(new_sid)
                    rec["emb"] = None
                    rec["last_emb_ts"] = 0.0
                    rec["pose_vec"] = None
                    rec["last_pose_ts"] = 0.0
                    rec["identity_quality"] = "weak_no_embedding"
                    rec["first_seen_ts"] = float(ts)
                    rec["early_emb_reconcile_attempts"] = 0
                    # Reset confirm counters so the split track must re-confirm.
                    self._pending_new_counts[key] = 0
                    self._pending_new_ts[key] = float(ts)
                    self._household_emb_counts[key] = 0
                    self.active_zones[int(new_sid)].add((int(sensor_id), zone_existing))
                    # If the abandoned SID was provisional and nobody holds it, free it.
                    if self.household_mode and self._household_is_provisional_sid(int(old_sid)):
                        still_used = any(
                            int(r.get("stable_id", -1)) == int(old_sid)
                            for r in self.active_tracks.values()
                        )
                        if not still_used:
                            self._household_release_provisional(int(old_sid))
                    diag_event = "split_same_frame_conflict"
                    diag_reject_reason = "same_frame_sid_conflict"
                    diag_sid_candidate = int(old_sid)

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
            if is_new and emb is None:
                self._sid_no_embedding_count += 1
                diag_reject_reason = "no_embedding"

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
                identity_kind: Optional[str] = None
                # Prefer ghost match (same camera, recent disappearance)
                if emb is not None or pose_valid:
                    sid = self._match_ghost(
                        sensor_id,
                        emb,
                        bbox_ltrbwh,
                        ts,
                        pose_vec=pose_vec if pose_valid else None,
                        world_xy=world_xy,
                        world_valid=bool(world_valid),
                    )
                    if sid is not None:
                        diag_event = "match_ghost"
                        diag_sid_candidate = int(sid)
                    # Cross-camera active/gallery match if enabled
                    if sid is None and emb is not None:
                        g_kind = "visitor"
                        if self.household_mode:
                            g_id, g_reid, g_req, g_kind = self._household_gallery_best(
                                emb,
                                sensor_id=int(sensor_id),
                                curr_bbox=bbox_ltrbwh,
                                curr_brightness=curr_brightness,
                                curr_color=curr_color,
                                pose_vec=pose_vec if pose_valid else None,
                                pose_valid=pose_valid,
                                now_ts=float(ts),
                            )
                        else:
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
                        if g_id is not None:
                            diag_sid_candidate = int(g_id)
                        if g_id is not None and float(g_reid) >= float(g_req):
                            diag_reid_confidence = float(g_reid)
                            diag_reid_required = float(g_req)
                            ok, reason = self._gallery_match_ok(
                                track_key=key,
                                emb=emb,
                                g_id=int(g_id),
                                g_reid=float(g_reid),
                                g_req=float(g_req),
                                sensor_id=int(sensor_id),
                                bbox=bbox_ltrbwh,
                                ts=float(ts),
                                world_xy=world_xy,
                                world_valid=bool(world_valid),
                            )
                            if self._other_active_sensors_for_sid(int(g_id), int(sensor_id)):
                                diag_overlap_permit = bool(reason != "exclusivity_blocked")
                            if ok:
                                sid = g_id
                                diag_event = "match_gallery"
                                identity_kind = str(g_kind)
                            else:
                                diag_reject_reason = reason
                        elif g_id is not None:
                            diag_reject_reason = "threshold"
                    if sid is None and emb is None and pose_valid and pose_vec is not None:
                        g_id, g_pose, g_req = self._pose_gallery_best(pose_vec, now_ts=float(ts))
                        if g_id is not None and self.aliases_enabled:
                            g_id = self.canonical_sid(int(g_id))
                        if g_id is not None:
                            diag_sid_candidate = int(g_id)
                        if g_id is not None and float(g_pose) >= float(g_req):
                            diag_reid_confidence = float(g_pose)
                            diag_reid_required = float(g_req)
                            ok, reason = self._gallery_match_ok(
                                track_key=key,
                                emb=None,
                                g_id=int(g_id),
                                g_reid=float(g_pose),
                                g_req=float(g_req),
                                sensor_id=int(sensor_id),
                                bbox=bbox_ltrbwh,
                                ts=float(ts),
                                world_xy=world_xy,
                                world_valid=bool(world_valid),
                                require_mnn_emb=False,
                            )
                            if self._other_active_sensors_for_sid(int(g_id), int(sensor_id)):
                                diag_overlap_permit = bool(reason != "exclusivity_blocked")
                            if ok:
                                sid = g_id
                                diag_event = "match_pose_gallery"
                            else:
                                diag_reject_reason = reason
                        elif g_id is not None:
                            diag_reject_reason = "threshold"

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
                    household_confirmed = (
                        self._household_confirmed(key, emb) if self.household_mode else (cnt >= required)
                    )
                    if cnt < required or (self.household_mode and not household_confirmed):
                        # Defer finalizing permanent ID; return provisional/ephemeral ID for OSD continuity.
                        pending_sid = self._pending_new_sids.get(key)
                        first_mint = pending_sid is None
                        if pending_sid is None:
                            if self.household_mode:
                                pending_sid = int(self._household_alloc_provisional(exclude_key=key))
                                self._provisional_event_count += 1
                            else:
                                pending_sid = int(self._alloc_sid())
                            self._pending_new_sids[key] = int(pending_sid)
                            self._sid_new_alloc_count += 1
                            if active_ids_here:
                                self._sid_fragmentation_events += 1
                        if self.household_mode:
                            diag_event = "provisional"
                            diag_sid_candidate = int(pending_sid)
                            if first_mint:
                                self._record_id_event(diag_event)
                            id_kind = "provisional"
                            diag_payload = self._set_track_diag(
                                key,
                                event=diag_event,
                                reject_reason=diag_reject_reason,
                                embedding_present=emb is not None,
                                pose_present=pose_valid and pose_vec is not None,
                                sid_candidate=diag_sid_candidate,
                                stable_id=int(pending_sid),
                                ts=float(ts),
                                reid_confidence=diag_reid_confidence,
                                reid_required=diag_reid_required,
                                overlap_permit=diag_overlap_permit,
                                identity_kind=id_kind,
                                identity_state=identity_state_for_kind(id_kind),
                            )
                            # Create a full active provisional record so subsequent frames
                            # hit is_new=False, exclusivity sees the claim, and SID stays stable.
                            rec = {
                                "stable_id": int(pending_sid),
                                "first_seen_ts": float(ts),
                                "bbox": bbox_ltrbwh,
                                "last_seen_ts": float(ts),
                                "last_emb_ts": float(ts) if emb is not None else 0.0,
                                "emb": emb,
                                "pose_vec": pose_vec,
                                "last_pose_ts": float(ts) if pose_vec is not None else 0.0,
                                "early_emb_reconcile_attempts": 0,
                                "zone": zone or "default",
                                "id_diag": diag_payload,
                                "identity_quality": (
                                    "strong"
                                    if emb is not None
                                    else ("pose_only" if pose_vec is not None else "weak_no_embedding")
                                ),
                            }
                            self._household_apply_identity_meta(rec, sid=int(pending_sid), kind="provisional")
                            self.active_tracks[key] = rec
                            self.active_zones[int(pending_sid)].add((int(sensor_id), rec["zone"]))
                            # Ownership moves to active_tracks; keep confirm counters, drop pending SID map.
                            self._pending_new_sids.pop(key, None)
                            self._record_sid_sensor_world(
                                int(pending_sid),
                                int(sensor_id),
                                world_xy=world_xy,
                                world_valid=bool(world_valid),
                                ts=float(ts),
                            )
                            if self.aliases_enabled:
                                return int(self.canonical_sid(int(pending_sid)))
                            return int(pending_sid)
                        diag_event = "new_alloc"
                        diag_sid_candidate = int(pending_sid)
                        self._record_id_event(diag_event)
                        id_kind = self._household_identity_kind(int(pending_sid))
                        self._set_track_diag(
                            key,
                            event=diag_event,
                            reject_reason=diag_reject_reason,
                            embedding_present=emb is not None,
                            pose_present=pose_valid and pose_vec is not None,
                            sid_candidate=diag_sid_candidate,
                            stable_id=int(pending_sid),
                            ts=float(ts),
                            reid_confidence=diag_reid_confidence,
                            reid_required=diag_reid_required,
                            overlap_permit=diag_overlap_permit,
                            identity_kind=id_kind,
                            identity_state=identity_state_for_kind(id_kind),
                        )
                        if self.aliases_enabled:
                            return int(self.canonical_sid(int(pending_sid)))
                        return int(pending_sid)
                    # Confirmation reached
                    pending_sid = self._pending_new_sids.get(key)
                    identity_kind = "visitor"
                    if self.household_mode:
                        sid, identity_kind, diag_event = self._household_finalize_new_sid(
                            key,
                            emb=emb,
                            sensor_id=int(sensor_id),
                            bbox_ltrbwh=bbox_ltrbwh,
                            ts=float(ts),
                            curr_brightness=curr_brightness,
                            curr_color=curr_color,
                            pose_vec=pose_vec,
                            pose_valid=pose_valid,
                            world_xy=world_xy,
                            world_valid=bool(world_valid),
                            pending_sid=int(pending_sid) if pending_sid is not None else None,
                        )
                        self._sid_new_alloc_count += 1
                        if active_ids_here:
                            self._sid_fragmentation_events += 1
                    else:
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
                                    if self._allow_cross_camera_sid_active(
                                        int(g_id2),
                                        int(sensor_id),
                                        world_xy=world_xy,
                                        world_valid=bool(world_valid),
                                        ts=float(ts),
                                        appearance_sim=float(g_reid2),
                                    ):
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
                            self._sid_new_alloc_count += 1
                            if active_ids_here:
                                self._sid_fragmentation_events += 1
                        diag_event = "new_alloc"
                    # Clear pending counter after minting
                    try:
                        self._clear_pending_new_state(key, keep_sid=int(sid))
                    except Exception:
                        pass

                # Creating an active record: clear any pending state for this track key.
                try:
                    self._clear_pending_new_state(key, keep_sid=int(sid))
                except Exception:
                    pass

                if self.aliases_enabled:
                    sid = self.canonical_sid(int(sid))
                diag_sid_candidate = int(sid)
                diag_payload = self._set_track_diag(
                    key,
                    event=diag_event,
                    reject_reason=diag_reject_reason,
                    embedding_present=emb is not None,
                    pose_present=pose_valid and pose_vec is not None,
                    sid_candidate=diag_sid_candidate,
                    stable_id=int(sid),
                    ts=float(ts),
                    reid_confidence=diag_reid_confidence,
                    reid_required=diag_reid_required,
                    overlap_permit=diag_overlap_permit,
                )
                self._record_id_event(diag_event)

                rec = {
                    "stable_id": int(sid),
                    "first_seen_ts": float(ts),
                    "bbox": bbox_ltrbwh,
                    "last_seen_ts": float(ts),
                    "last_emb_ts": float(ts) if emb is not None else 0.0,
                    "emb": emb,
                    "pose_vec": pose_vec,
                    "last_pose_ts": float(ts) if pose_vec is not None else 0.0,
                    "early_emb_reconcile_attempts": 0,
                    "zone": zone or "default",
                    "id_diag": diag_payload,
                    "identity_quality": (
                        "strong"
                        if emb is not None
                        else ("pose_only" if pose_vec is not None else "weak_no_embedding")
                    ),
                }
                if self.household_mode:
                    kind = identity_kind if identity_kind is not None else self._household_identity_kind(int(sid))
                    self._household_apply_identity_meta(rec, sid=int(sid), kind=str(kind))
                    self._household_touch_visitor(int(sid), float(ts))
                self.active_tracks[key] = rec
                # Track zones and gallery
                self.active_zones[int(sid)].add((int(sensor_id), rec["zone"]))
                if emb is not None:
                    sid_int = int(sid)
                    self._persist_track_embedding(
                        sid_int,
                        float(ts),
                        emb,
                        rec=rec,
                        bbox=bbox_ltrbwh,
                        pose_quality=pose_quality,
                    )
                    if sid_int not in self.sid_global_first_seen:
                        self.sid_global_first_seen[sid_int] = float(ts)
                    # Initialize EMA centroid with first embedding
                    try:
                        newc = emb / (np.linalg.norm(emb) + 1e-12)
                        self.sid_centroid[sid_int] = newc.astype(np.float32)
                    except Exception:
                        pass
                    self.sid_last_bbox[sid_int] = bbox_ltrbwh
                    self.sid_last_bbox_sensor[sid_int] = int(sensor_id)
                    if curr_brightness is not None:
                        self.sid_last_brightness[sid_int] = curr_brightness
                    if curr_color is not None:
                        self.sid_last_color[sid_int] = curr_color.astype(np.float32)
                if pose_vec is not None:
                    self._update_pose_state(int(sid), pose_vec, float(ts))
                self._record_sid_sensor_world(
                    int(sid),
                    int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    ts=float(ts),
                )
                self._auto_merge_if_needed(float(ts))
                try:
                    latest = self.active_tracks.get(key, rec)
                    sid_out = int(latest.get("stable_id", sid))
                except Exception:
                    sid_out = int(sid)
                if self.aliases_enabled:
                    sid_out = int(self.canonical_sid(sid_out))
                return int(sid_out)

            # Existing track: update
            rec["bbox"] = bbox_ltrbwh
            rec["last_seen_ts"] = float(ts)
            if zone:
                rec["zone"] = zone
                self.active_zones[int(rec["stable_id"])].add((int(sensor_id), rec["zone"]))

            # Household provisional: accumulate confirm evidence and promote in place.
            if self.household_mode and str(rec.get("identity_kind") or "") == "provisional":
                cnt = int(self._pending_new_counts.get(key, 0)) + 1
                self._pending_new_counts[key] = cnt
                self._pending_new_ts[key] = float(ts)
                if emb is not None:
                    rec["emb"] = emb
                    rec["last_emb_ts"] = float(ts)
                    rec["identity_quality"] = "strong"
                if pose_vec is not None and pose_valid:
                    rec["pose_vec"] = pose_vec
                    rec["last_pose_ts"] = float(ts)
                    if rec.get("identity_quality") in (None, "", "weak_no_embedding"):
                        rec["identity_quality"] = "pose_only"
                if self._household_confirmed(key, emb):
                    old_sid = int(rec.get("stable_id"))
                    new_sid, identity_kind, diag_event = self._household_finalize_new_sid(
                        key,
                        emb=emb if emb is not None else rec.get("emb"),
                        sensor_id=int(sensor_id),
                        bbox_ltrbwh=bbox_ltrbwh,
                        ts=float(ts),
                        curr_brightness=curr_brightness,
                        curr_color=curr_color,
                        pose_vec=pose_vec if pose_valid else rec.get("pose_vec"),
                        pose_valid=bool(pose_valid or rec.get("pose_vec") is not None),
                        world_xy=world_xy,
                        world_valid=bool(world_valid),
                        pending_sid=int(old_sid),
                    )
                    if int(new_sid) != int(old_sid):
                        self._remap_sid(
                            int(old_sid),
                            int(new_sid),
                            kind=str(identity_kind),
                            purge_old=True,
                            move_gallery=False,
                        )
                    else:
                        self._household_apply_identity_meta(rec, sid=int(new_sid), kind=str(identity_kind))
                    self._household_touch_visitor(int(new_sid), float(ts))
                    self._clear_pending_new_state(key, keep_sid=int(new_sid))
                    emb_for_persist = emb if emb is not None else rec.get("emb")
                    if emb_for_persist is not None:
                        self._persist_track_embedding(
                            int(new_sid),
                            float(ts),
                            emb_for_persist,
                            rec=rec,
                            bbox=bbox_ltrbwh,
                            pose_quality=pose_quality,
                        )
                        try:
                            newc = emb_for_persist / (np.linalg.norm(emb_for_persist) + 1e-12)
                            self.sid_centroid[int(new_sid)] = newc.astype(np.float32)
                        except Exception:
                            pass
                        self.sid_last_bbox[int(new_sid)] = bbox_ltrbwh
                        self.sid_last_bbox_sensor[int(new_sid)] = int(sensor_id)
                    if pose_vec is not None and pose_valid:
                        self._update_pose_state(int(new_sid), pose_vec, float(ts))
                    diag_payload = self._set_track_diag(
                        key,
                        event=str(diag_event),
                        reject_reason=diag_reject_reason,
                        embedding_present=emb_for_persist is not None,
                        pose_present=(pose_valid and pose_vec is not None) or rec.get("pose_vec") is not None,
                        sid_candidate=int(new_sid),
                        stable_id=int(new_sid),
                        ts=float(ts),
                        reid_confidence=diag_reid_confidence,
                        reid_required=diag_reid_required,
                        overlap_permit=diag_overlap_permit,
                        identity_kind=str(identity_kind),
                        identity_state=identity_state_for_kind(str(identity_kind)),
                    )
                    self._record_id_event(str(diag_event))
                    rec["id_diag"] = diag_payload
                    self.active_tracks[key] = rec
                    self._record_sid_sensor_world(
                        int(new_sid),
                        int(sensor_id),
                        world_xy=world_xy,
                        world_valid=bool(world_valid),
                        ts=float(ts),
                    )
                    if self.aliases_enabled:
                        return int(self.canonical_sid(int(new_sid)))
                    return int(new_sid)
                # Still provisional: refresh diag and return same SID (no gallery write).
                diag_payload = self._set_track_diag(
                    key,
                    event="provisional",
                    reject_reason=diag_reject_reason,
                    embedding_present=rec.get("emb") is not None,
                    pose_present=rec.get("pose_vec") is not None,
                    sid_candidate=int(rec.get("stable_id")),
                    stable_id=int(rec.get("stable_id")),
                    ts=float(ts),
                    reid_confidence=diag_reid_confidence,
                    reid_required=diag_reid_required,
                    overlap_permit=diag_overlap_permit,
                    identity_kind="provisional",
                    identity_state="provisional",
                )
                rec["id_diag"] = diag_payload
                self.active_tracks[key] = rec
                self._record_sid_sensor_world(
                    int(rec["stable_id"]),
                    int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    ts=float(ts),
                )
                if self.aliases_enabled:
                    return int(self.canonical_sid(int(rec["stable_id"])))
                return int(rec["stable_id"])

            if emb is not None:
                # Reconcile weak/provisional IDs when fresh embeddings arrive.
                # Besides the "no embedding yet" path, allow a short early-reconcile
                # window for low-support IDs to reduce needless ID fragmentation.
                rec_emb_missing = rec.get("emb") is None
                try:
                    current_sid = int(rec.get("stable_id"))
                except Exception:
                    current_sid = int(rec["stable_id"])
                sid_support = int(len(self.gallery.get(int(current_sid), [])))
                try:
                    first_seen_ts = float(rec.get("first_seen_ts", rec.get("last_seen_ts", ts)))
                except Exception:
                    first_seen_ts = float(ts)
                track_age_s = max(0.0, float(ts) - float(first_seen_ts))
                try:
                    early_attempts = int(rec.get("early_emb_reconcile_attempts", 0) or 0)
                except Exception:
                    early_attempts = 0
                allow_early_reconcile = (
                    (not rec_emb_missing)
                    and sid_support <= int(self.early_reconcile_max_support)
                    and track_age_s <= float(self.early_reconcile_window_s)
                    and early_attempts < int(self.early_reconcile_max_attempts)
                )
                if rec_emb_missing or allow_early_reconcile:
                    if allow_early_reconcile:
                        rec["early_emb_reconcile_attempts"] = int(early_attempts) + 1
                    candidate_sid = None
                    try:
                        candidate_sid = self._match_ghost(
                            sensor_id,
                            emb,
                            bbox_ltrbwh,
                            ts,
                            pose_vec=pose_vec if pose_valid else None,
                            exclude_key=key,
                            world_xy=world_xy,
                            world_valid=bool(world_valid),
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
                        if g_id is not None:
                            diag_sid_candidate = int(g_id)
                        if g_id is not None and float(g_reid) >= float(g_req):
                            diag_reid_confidence = float(g_reid)
                            diag_reid_required = float(g_req)
                            ok, reason = self._gallery_match_ok(
                                track_key=key,
                                emb=emb,
                                g_id=int(g_id),
                                g_reid=float(g_reid),
                                g_req=float(g_req),
                                sensor_id=int(sensor_id),
                                bbox=bbox_ltrbwh,
                                ts=float(ts),
                                world_xy=world_xy,
                                world_valid=bool(world_valid),
                                exclude_key=key,
                            )
                            if self._other_active_sensors_for_sid(int(g_id), int(sensor_id)):
                                diag_overlap_permit = bool(reason != "exclusivity_blocked")
                            if ok:
                                candidate_sid = int(g_id)
                            else:
                                diag_reject_reason = reason
                        elif g_id is not None:
                            diag_reject_reason = "threshold"
                    if candidate_sid is not None and self.aliases_enabled:
                        candidate_sid = self.canonical_sid(int(candidate_sid))
                    if candidate_sid is not None and int(candidate_sid) > 0 and int(candidate_sid) != int(current_sid):
                        diag_event = "remap_on_first_emb" if rec_emb_missing else "remap_on_early_emb"
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
                        self._sid_remap_count += 1

                rec["emb"] = emb
                rec["last_emb_ts"] = float(ts)
                rec["identity_quality"] = "strong"
                sid_int = int(rec["stable_id"]) 
                self._persist_track_embedding(
                    sid_int,
                    float(ts),
                    emb,
                    rec=rec,
                    bbox=bbox_ltrbwh,
                    pose_quality=pose_quality,
                )
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
                self.sid_last_bbox_sensor[sid_int] = int(sensor_id)
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
                            exclude_key=key,
                            world_xy=world_xy,
                            world_valid=bool(world_valid),
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
                        if g_id is not None:
                            diag_sid_candidate = int(g_id)
                        if g_id is not None and float(g_pose) >= float(g_req):
                            diag_reid_confidence = float(g_pose)
                            diag_reid_required = float(g_req)
                            ok, reason = self._gallery_match_ok(
                                track_key=key,
                                emb=None,
                                g_id=int(g_id),
                                g_reid=float(g_pose),
                                g_req=float(g_req),
                                sensor_id=int(sensor_id),
                                bbox=bbox_ltrbwh,
                                ts=float(ts),
                                world_xy=world_xy,
                                world_valid=bool(world_valid),
                                exclude_key=key,
                                require_mnn_emb=False,
                            )
                            if self._other_active_sensors_for_sid(int(g_id), int(sensor_id)):
                                diag_overlap_permit = bool(reason != "exclusivity_blocked")
                            if ok:
                                candidate_sid = int(g_id)
                            else:
                                diag_reject_reason = reason
                        elif g_id is not None:
                            diag_reject_reason = "threshold"
                    if candidate_sid is not None and self.aliases_enabled:
                        candidate_sid = self.canonical_sid(int(candidate_sid))
                    if candidate_sid is not None and int(candidate_sid) > 0 and int(candidate_sid) != int(current_sid):
                        diag_event = "remap_on_first_pose"
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
                        self._sid_remap_count += 1
                rec["pose_vec"] = pose_vec
                rec["last_pose_ts"] = float(ts)
                if rec.get("identity_quality") in (None, "", "weak_no_embedding"):
                    rec["identity_quality"] = "pose_only"
                sid_int = int(rec["stable_id"])
                self._update_pose_state(sid_int, pose_vec, float(ts))
            diag_payload = self._set_track_diag(
                key,
                event=diag_event,
                reject_reason=diag_reject_reason,
                embedding_present=emb is not None or rec.get("emb") is not None,
                pose_present=(pose_valid and pose_vec is not None) or rec.get("pose_vec") is not None,
                sid_candidate=diag_sid_candidate,
                stable_id=int(rec.get("stable_id")),
                ts=float(ts),
                reid_confidence=diag_reid_confidence,
                reid_required=diag_reid_required,
                overlap_permit=diag_overlap_permit,
            )
            self._record_id_event(diag_event)
            rec["id_diag"] = diag_payload
            self.active_tracks[key] = rec
            try:
                self.sid_global_last_seen[int(rec["stable_id"])] = float(ts)
            except Exception:
                pass
            self._record_sid_sensor_world(
                int(rec["stable_id"]),
                int(sensor_id),
                world_xy=world_xy,
                world_valid=bool(world_valid),
                ts=float(ts),
            )
            self._maybe_autosave_gallery(float(ts))
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
                    self._clear_pending_new_state(k)
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
                self._pending_id_diag.pop(key, None)
                self._pending_new_counts.pop(key, None)
                self._pending_new_ts.pop(key, None)
                self._household_emb_counts.pop(key, None)
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
                kind = str(rec.get("identity_kind") or "")
                is_provisional = kind == "provisional" or self._household_is_provisional_sid(sid)
                if (emb_val is not None or pose_val is not None) and not is_provisional:
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
                elif is_provisional:
                    still_used = any(
                        int(r.get("stable_id", -1)) == sid for r in self.active_tracks.values()
                    )
                    if not still_used:
                        self._household_release_provisional(sid)

    def prune_ghosts(self, now_ts: Optional[float] = None) -> None:
        with self._lock:
            t = float(now_ts if now_ts is not None else time.time())
            for sensor_id, dq in self.ghosts.items():
                while dq and (t - float(dq[0].get("ts", 0.0))) > self.max_ghost_age_s:
                    dq.popleft()
            # Legacy mode may recycle fully inactive SIDs after a cooldown.
            # Household residents are durable enrollments and visitors have a
            # separate TTL pool below, so the generic purge must never run for
            # household state.
            if not self.household_mode:
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
                        self._clear_pending_new_state(key)
            except Exception:
                pass
            try:
                self._prune_pose_gallery(t)
            except Exception:
                pass
            if self.household_mode and self._visitor_pool is not None:
                try:
                    active_sids = {int(rec.get("stable_id")) for rec in self.active_tracks.values()}
                    ghost_sids = set()
                    for dq in self.ghosts.values():
                        for g in dq:
                            try:
                                ghost_sids.add(int(g.get("stable_id")))
                            except Exception:
                                pass
                    recycled = self._visitor_pool.recycle_inactive(
                        float(t),
                        active_sids=active_sids,
                        ghost_sids=ghost_sids,
                    )
                    if recycled:
                        for sid in recycled:
                            if sid not in active_sids and sid not in ghost_sids:
                                self._purge_sid_state(int(sid))
                        self._visitor_pool.save()
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
                    "canonical_gallery_size": int(self._count_canonical_sids()),
                    "auto_merge_enabled": bool(self.auto_merge_enabled),
                    "auto_merge_interval_s": float(self.auto_merge_interval_s),
                    "auto_merge_allow_both_active": bool(self.auto_merge_allow_both_active),
                    "auto_merge_both_active_min_sim": float(self.auto_merge_both_active_min_sim),
                    "auto_merge_runs": int(self._auto_merge_runs),
                    "auto_merge_triggers": int(self._auto_merge_triggers),
                    "auto_merge_applied": int(self._auto_merge_applied),
                    "auto_merge_suppressed": int(self._auto_merge_suppressed),
                    "auto_merge_last_candidate_count": int(self._auto_merge_last_candidate_count),
                    "auto_merge_last_blocked_count": int(self._auto_merge_last_blocked_count),
                    "auto_merge_last_support_gate": int(self._auto_merge_last_support_gate),
                    "auto_merge_zero_apply_streak": int(self._auto_merge_zero_apply_streak),
                    "auto_merge_last_ts": float(self._last_auto_merge_ts) if self._last_auto_merge_ts is not None else None,
                    "auto_merge_last_reason": str(self._auto_merge_last_reason),
                    "auto_merge_last_pressure_recycled": int(self._auto_merge_last_pressure_recycled),
                    "auto_merge_pressure_recycled_total": int(self._auto_merge_pressure_recycled_total),
                    "sid_new_alloc_count": int(self._sid_new_alloc_count),
                    "sid_remap_count": int(self._sid_remap_count),
                    "sid_guard_reject_count": int(self._sid_guard_reject_count),
                    "sid_no_embedding_count": int(self._sid_no_embedding_count),
                    "sid_fragmentation_events": int(self._sid_fragmentation_events),
                    "sid_pending_recycled_count": int(self._sid_pending_recycled_count),
                    "sid_same_frame_conflict_count": int(self._sid_same_frame_conflict_count),
                    "false_share_blocked_count": int(self._false_share_blocked_count),
                    "overlap_permit_grant_count": int(self._overlap_permit_grant_count),
                    "overlap_permit_deny_count": int(self._overlap_permit_deny_count),
                    "overlap_permit_degraded_count": int(self._overlap_permit_degraded_count),
                    "gallery_quality_reject_count": int(self._gallery_quality_reject_count),
                    "mnn_reject_count": int(self._mnn_reject_count),
                    "household_mode": bool(self.household_mode),
                    # provisional_count = currently active provisional tracks
                    "provisional_count": int(self._count_active_provisional()) if self.household_mode else 0,
                    # provisional_event_count = cumulative first-mint provisional events
                    "provisional_event_count": int(self._provisional_event_count),
                    "provisional_active_count": int(self._count_active_provisional()) if self.household_mode else 0,
                    "provisional_pool_used": (
                        int(self._provisional_pool.used_count())
                        if self.household_mode and self._provisional_pool is not None
                        else 0
                    ),
                    "provisional_pool_free": (
                        int(self._provisional_pool.free_count())
                        if self.household_mode and self._provisional_pool is not None
                        else 0
                    ),
                    "resident_count": int(len(self._household_resident_ids())) if self.household_mode else 0,
                    "visitor_count": int(
                        sum(
                            1
                            for sid in self.gallery.keys()
                            if self.visitor_id_min <= int(sid) <= self.visitor_id_max
                        )
                    )
                    if self.household_mode
                    else 0,
                    "mint_visitor_count": int(self._mint_visitor_count),
                    "promote_resident_count": int(self._promote_resident_count),
                    "camera_topology_file": self._camera_topology.source_path,
                    "camera_topology_validation_status": str(self._camera_topology_validation_status),
                    "camera_topology_validation_errors": list(self._camera_topology_validation_errors),
                    "visitor_pool_used": (
                        len(self._visitor_pool.used_sids())
                        if self.household_mode and self._visitor_pool is not None
                        else 0
                    ),
                    "sid_event_counts": dict(self._id_event_counts),
                    "alias_candidate_pool_size": int(self._alias_candidate_pool_size),
                    "alias_candidate_sim_p50": self._alias_candidate_sim_p50,
                    "alias_candidate_sim_p95": self._alias_candidate_sim_p95,
                    "alias_support_ids_ge_1": int(self._alias_support_ids_ge_1),
                    "alias_support_ids_ge_2": int(self._alias_support_ids_ge_2),
                    "alias_support_ids_ge_3": int(self._alias_support_ids_ge_3),
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
        exclude_key: Optional[Tuple[int, int]] = None,
        world_xy: Optional[Tuple[float, float]] = None,
        world_valid: bool = False,
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
            if self._sid_claimed_same_frame(
                sensor_id=int(sensor_id),
                sid=int(g_sid),
                ts=float(ts),
                exclude_key=exclude_key,
            ):
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
                if not self._allow_cross_camera_sid_active(
                    int(g_sid),
                    int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    ts=float(ts),
                    appearance_sim=float(sim),
                ):
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
                if not self._allow_cross_camera_sid_active(
                    int(g_sid),
                    int(sensor_id),
                    world_xy=world_xy,
                    world_valid=bool(world_valid),
                    ts=float(ts),
                    appearance_sim=float(pose_sim),
                ):
                    continue
                score = float(pose_sim)
                if score > best_score:
                    best_sid, best_score = g_sid, score
        return best_sid
