"""Global assignment helpers (mutual-nearest / frame exclusivity) for household mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

TrackKey = Tuple[int, int]
SimilarityFn = Callable[[np.ndarray, List[int]], Dict[int, float]]


@dataclass
class FrameAssignmentState:
    """Tracks SID claims within a coalesced resolve frame."""

    epsilon_s: float = 1e-2
    _frame_ts: Optional[float] = None
    _claims: Dict[int, TrackKey] = field(default_factory=dict)

    def _same_frame(self, ts: float) -> bool:
        if self._frame_ts is None:
            return False
        return abs(float(ts) - float(self._frame_ts)) <= float(self.epsilon_s)

    def reset_if_new_frame(self, ts: float) -> None:
        if not self._same_frame(ts):
            self._frame_ts = float(ts)
            self._claims.clear()

    def is_claimed(self, sid: int, *, exclude_key: Optional[TrackKey] = None) -> bool:
        key = self._claims.get(int(sid))
        if key is None:
            return False
        if exclude_key is not None and key == exclude_key:
            return False
        return True

    def claim(self, sid: int, track_key: TrackKey, ts: float) -> None:
        self.reset_if_new_frame(ts)
        self._claims[int(sid)] = track_key

    def claimed_by(self, sid: int) -> Optional[TrackKey]:
        return self._claims.get(int(sid))


def active_embeddings_same_frame(
    active_tracks: Dict[TrackKey, dict],
    ts: float,
    *,
    epsilon_s: float,
    exclude_key: Optional[TrackKey] = None,
) -> Dict[TrackKey, np.ndarray]:
    out: Dict[TrackKey, np.ndarray] = {}
    for key, rec in active_tracks.items():
        if exclude_key is not None and key == exclude_key:
            continue
        emb = rec.get("emb")
        if emb is None:
            continue
        try:
            last_ts = float(rec.get("last_seen_ts", -1e18))
        except Exception:
            continue
        if abs(last_ts - float(ts)) > float(epsilon_s):
            continue
        try:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            if vec.size < 1:
                continue
            out[key] = vec
        except Exception:
            continue
    return out


def best_sid_for_embedding(
    emb: np.ndarray,
    candidate_sids: List[int],
    sim_fn: SimilarityFn,
) -> Tuple[Optional[int], float]:
    if not candidate_sids:
        return None, -1.0
    sims = sim_fn(emb, candidate_sids)
    if not sims:
        return None, -1.0
    best_sid = max(sims, key=lambda s: float(sims[s]))
    return int(best_sid), float(sims[best_sid])


def is_mutual_nearest_match(
    query_key: TrackKey,
    query_emb: np.ndarray,
    best_sid: int,
    peer_embeddings: Dict[TrackKey, np.ndarray],
    candidate_sids: List[int],
    sim_fn: SimilarityFn,
) -> bool:
    """True when best_sid's top query this frame is query_key (MNN)."""
    if int(best_sid) <= 0:
        return False
    sid_best_key: Optional[TrackKey] = None
    sid_best_score = -1.0
    for peer_key, peer_emb in peer_embeddings.items():
        sims = sim_fn(peer_emb, [int(best_sid)])
        score = float(sims.get(int(best_sid), -1.0))
        if score > sid_best_score:
            sid_best_score = score
            sid_best_key = peer_key
    if sid_best_key is None:
        return True
    return sid_best_key == query_key


def assign_tracklets_mnn(
    track_embeddings: Dict[TrackKey, np.ndarray],
    candidate_sids: List[int],
    sim_fn: SimilarityFn,
    *,
    min_score: float = 0.0,
) -> Dict[TrackKey, Tuple[int, float]]:
    """Greedy mutual-nearest assignment for a batch of tracklets vs identities."""
    if not track_embeddings or not candidate_sids:
        return {}

    remaining_tracks = set(track_embeddings.keys())
    remaining_sids = set(int(s) for s in candidate_sids)
    assignments: Dict[TrackKey, Tuple[int, float]] = {}

    while remaining_tracks and remaining_sids:
        best_pair: Optional[Tuple[TrackKey, int, float]] = None
        for t_key in list(remaining_tracks):
            emb = track_embeddings[t_key]
            sims = sim_fn(emb, list(remaining_sids))
            if not sims:
                continue
            sid = max(sims, key=lambda s: float(sims[s]))
            score = float(sims[int(sid)])
            if score < float(min_score):
                continue
            if best_pair is None or score > best_pair[2]:
                best_pair = (t_key, int(sid), score)

        if best_pair is None:
            break

        t_key, sid, score = best_pair
        peers = {k: v for k, v in track_embeddings.items() if k in remaining_tracks}
        if not is_mutual_nearest_match(t_key, track_embeddings[t_key], sid, peers, list(remaining_sids), sim_fn):
            remaining_tracks.discard(t_key)
            continue

        assignments[t_key] = (int(sid), float(score))
        remaining_tracks.discard(t_key)
        remaining_sids.discard(int(sid))

    return assignments
