"""Quality gates and clustered exemplar storage for identity galleries."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]

# Cosine similarity below this → treat as a distinct appearance cluster.
_DEFAULT_CLUSTER_MIN_SIM = 0.75
# Replace near-duplicate exemplars above this similarity.
_DEFAULT_REPLACE_MIN_SIM = 0.92


def _normalize(emb: np.ndarray) -> np.ndarray:
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    return vec / (float(np.linalg.norm(vec)) + 1e-12)


def bbox_crop_height(bbox: Optional[BBox]) -> Optional[float]:
    if bbox is None or len(bbox) < 4:
        return None
    try:
        return float(bbox[3])
    except Exception:
        return None


def bbox_aspect_ok(
    bbox: Optional[BBox],
    *,
    min_ratio: float = 0.15,
    max_ratio: float = 0.85,
) -> bool:
    """Reject extreme aspect crops (very wide or very tall boxes)."""
    if bbox is None or len(bbox) < 4:
        return True
    try:
        w = max(1.0, float(bbox[2]))
        h = max(1.0, float(bbox[3]))
        ratio = min(w / h, h / w)
        return float(min_ratio) <= ratio <= float(max_ratio)
    except Exception:
        return True


def pose_occlusion_heavy(pose_quality: Optional[dict]) -> bool:
    if not pose_quality:
        return False
    try:
        valid_frac = pose_quality.get("kpt_valid_frac", pose_quality.get("valid_frac"))
        mean_conf = pose_quality.get("kpt_mean_conf", pose_quality.get("mean_conf"))
        if valid_frac is not None and float(valid_frac) < 0.25:
            return True
        if mean_conf is not None and float(mean_conf) < 0.20:
            return True
    except Exception:
        return False
    return False


def gallery_identity_confirmed(
    *,
    identity_state: Optional[str],
    identity_quality: Optional[str],
    sid_has_gallery: bool,
) -> bool:
    """Gallery updates require a non-provisional identity (Phase 1 hook)."""
    if identity_state in ("resident", "visitor"):
        return True
    if identity_state == "provisional":
        return False
    if identity_quality in (None, "", "weak_no_embedding"):
        return not sid_has_gallery
    return True


def should_accept_gallery_embedding(
    *,
    household_mode: bool,
    bbox: Optional[BBox],
    blur_var: Optional[float],
    pose_quality: Optional[dict],
    identity_state: Optional[str],
    identity_quality: Optional[str],
    sid_has_gallery: bool,
    min_crop_h: int,
    min_laplacian_var: float,
) -> Tuple[bool, Optional[str]]:
    """Return (accept, reject_reason). Legacy mode keeps prior permissive behavior."""
    if not household_mode:
        return True, None

    crop_h = bbox_crop_height(bbox)
    if crop_h is not None and crop_h < float(min_crop_h):
        return False, "crop_too_small"

    if not bbox_aspect_ok(bbox):
        return False, "aspect_ratio"

    if blur_var is not None and float(blur_var) < float(min_laplacian_var):
        return False, "blur"

    if pose_occlusion_heavy(pose_quality):
        return False, "occlusion"

    if not gallery_identity_confirmed(
        identity_state=identity_state,
        identity_quality=identity_quality,
        sid_has_gallery=sid_has_gallery,
    ):
        return False, "provisional"

    return True, None


def add_clustered_exemplar(
    dq: Deque[Tuple[float, np.ndarray]],
    ts: float,
    emb: np.ndarray,
    *,
    cluster_min_sim: float = _DEFAULT_CLUSTER_MIN_SIM,
    replace_min_sim: float = _DEFAULT_REPLACE_MIN_SIM,
) -> None:
    """Insert embedding preserving diverse exemplar clusters (max len = deque maxlen)."""
    emb_n = _normalize(emb)
    if dq.maxlen is None or len(dq) < dq.maxlen:
        dq.append((float(ts), emb_n))
        return

    try:
        entries = list(dq)
        mat = np.stack([e for (_t, e) in entries], axis=0).astype(np.float32)
        sims = mat @ emb_n.reshape(-1)
        max_sim = float(np.max(sims))
        max_idx = int(np.argmax(sims))

        if max_sim >= float(replace_min_sim):
            entries[max_idx] = (float(ts), emb_n)
        elif max_sim < float(cluster_min_sim):
            # New appearance mode: evict the exemplar closest to the gallery mean.
            mean = mat.mean(axis=0)
            mean = mean / (float(np.linalg.norm(mean)) + 1e-12)
            mean_sims = mat @ mean.reshape(-1)
            redundant_idx = int(np.argmax(mean_sims))
            entries[redundant_idx] = (float(ts), emb_n)
        else:
            entries[max_idx] = (float(ts), emb_n)

        dq.clear()
        dq.extend(entries)
    except Exception:
        dq.append((float(ts), emb_n))
