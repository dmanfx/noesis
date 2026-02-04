from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Noesis DS8 ReID API")

_MANAGER_GETTER: Optional[Callable[[], Any]] = None


def register_reid_manager_getter(fn: Callable[[], Any]) -> None:
    global _MANAGER_GETTER
    _MANAGER_GETTER = fn


def _get_mgr() -> Any:
    if _MANAGER_GETTER is None:
        raise HTTPException(status_code=503, detail="ReID manager not registered")
    mgr = _MANAGER_GETTER()
    if mgr is None:
        raise HTTPException(status_code=503, detail="ReID manager unavailable")
    return mgr


def _list_aliases(mgr: Any) -> Dict[int, int]:
    if hasattr(mgr, "list_aliases") and callable(mgr.list_aliases):
        try:
            aliases = mgr.list_aliases()
        except Exception as exc:
            logger.warning("Failed to list aliases: %s", exc)
            return {}
        if isinstance(aliases, dict):
            return {int(k): int(v) for k, v in aliases.items()}
    return {}


def _merge_payload_to_response(
    payload: Dict[str, Any],
    *,
    fallback_a: Optional[int] = None,
    fallback_b: Optional[int] = None,
    aliases: Optional[Dict[int, int]] = None,
) -> "MergeResponse":
    src = payload.get("src", payload.get("a", fallback_a))
    dst = payload.get("dst", payload.get("b", fallback_b))
    canonical = payload.get("canonical")
    if canonical is None:
        canonical = payload.get("dst") or payload.get("b") or dst or src or fallback_a or 0
    return MergeResponse(
        applied=bool(payload.get("applied", False)),
        src=int(src or 0),
        dst=int(dst or 0),
        canonical=int(canonical or 0),
        reason=payload.get("reason"),
        aliases=aliases or {},
    )


class AliasListResponse(BaseModel):
    enabled: bool
    aliases: Dict[int, int]
    alias_file: Optional[str] = None
    copresence_window_s: float
    history_count: int


class MergeRequest(BaseModel):
    a: int = Field(..., ge=1)
    b: int = Field(..., ge=1)
    canonical: Optional[int] = Field(None, ge=1)
    append_embeddings: bool = True
    force: bool = False


class MergeResponse(BaseModel):
    applied: bool
    src: int
    dst: int
    canonical: int
    reason: Optional[str] = None
    aliases: Dict[int, int]


class MergeBatchRequest(BaseModel):
    pairs: List[MergeRequest] = Field(..., min_items=1)
    force: bool = False


class MergeBatchResponse(BaseModel):
    results: List[MergeResponse]
    applied_count: int
    failed_count: int
    aliases: Dict[int, int]


class UnsetRequest(BaseModel):
    src: int = Field(..., ge=1)


class UnsetResponse(BaseModel):
    src: int
    removed: bool
    reason: Optional[str] = None
    aliases: Dict[int, int]


class SuggestRequest(BaseModel):
    min_sim: Optional[float] = Field(None, ge=0.0, le=1.0)
    limit: int = Field(20, ge=1, le=500)
    require_inactive: bool = True


class SuggestCandidate(BaseModel):
    a: int
    b: int
    sim: float
    pose_sim: Optional[float] = None
    canonical: int
    preferred_canonical: int
    a_embedding_count: int
    b_embedding_count: int
    blocked: bool
    block_reason: Optional[str] = None


class SuggestResponse(BaseModel):
    candidates: List[SuggestCandidate]
    default_min_sim: float


@app.get(
    "/api/v1/reid/aliases",
    response_model=AliasListResponse,
    response_model_exclude_none=True,
)
def list_aliases() -> AliasListResponse:
    mgr = _get_mgr()
    aliases = _list_aliases(mgr)
    alias_file = getattr(mgr, "alias_file", None)
    history = getattr(mgr, "alias_history", [])
    try:
        history_count = int(len(history)) if isinstance(history, list) else 0
    except Exception:
        history_count = 0
    return AliasListResponse(
        enabled=bool(getattr(mgr, "aliases_enabled", False)),
        aliases=aliases,
        alias_file=str(alias_file) if alias_file else None,
        copresence_window_s=float(getattr(mgr, "copresence_window_s", 0.0)),
        history_count=history_count,
    )


@app.post(
    "/api/v1/reid/aliases/merge",
    response_model=MergeResponse,
    response_model_exclude_none=True,
)
def merge_aliases(req: MergeRequest) -> MergeResponse:
    mgr = _get_mgr()
    payload = mgr.set_alias(
        req.a,
        req.b,
        canonical=req.canonical,
        append_embeddings=req.append_embeddings,
        force=req.force,
        now_ts=time.time(),
    )
    aliases = _list_aliases(mgr)
    return _merge_payload_to_response(payload, fallback_a=req.a, fallback_b=req.b, aliases=aliases)


@app.post(
    "/api/v1/reid/aliases/merge-batch",
    response_model=MergeBatchResponse,
    response_model_exclude_none=True,
)
def merge_batch(req: MergeBatchRequest) -> MergeBatchResponse:
    mgr = _get_mgr()
    force = bool(req.force or any(pair.force for pair in req.pairs))
    pairs = [
        {
            "a": pair.a,
            "b": pair.b,
            "canonical": pair.canonical,
            "append_embeddings": pair.append_embeddings,
        }
        for pair in req.pairs
    ]
    payload = mgr.set_aliases_batch(pairs, force=force, now_ts=time.time())

    results_payload: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]] = None
    if isinstance(payload, dict):
        summary = payload
        results_payload = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    elif isinstance(payload, list):
        results_payload = payload
    else:
        results_payload = []

    aliases = _list_aliases(mgr)
    results = [
        _merge_payload_to_response(result, aliases=aliases)
        for result in results_payload
        if isinstance(result, dict)
    ]

    applied_count = None
    failed_count = None
    if summary:
        try:
            applied_count = int(summary.get("applied_count"))
        except Exception:
            applied_count = None
        try:
            failed_count = int(summary.get("failed_count"))
        except Exception:
            failed_count = None

    if applied_count is None:
        applied_count = sum(1 for result in results if result.applied)
    if failed_count is None:
        failed_count = max(0, len(results) - applied_count)

    return MergeBatchResponse(
        results=results,
        applied_count=applied_count,
        failed_count=failed_count,
        aliases=aliases,
    )


@app.post(
    "/api/v1/reid/aliases/unset",
    response_model=UnsetResponse,
    response_model_exclude_none=True,
)
def unset_alias(req: UnsetRequest) -> UnsetResponse:
    mgr = _get_mgr()
    payload = mgr.unset_alias(req.src)
    aliases = _list_aliases(mgr)
    return UnsetResponse(
        src=int(payload.get("src", req.src)),
        removed=bool(payload.get("removed", False)),
        reason=payload.get("reason"),
        aliases=aliases,
    )


@app.post(
    "/api/v1/reid/aliases/suggest",
    response_model=SuggestResponse,
    response_model_exclude_none=True,
)
def suggest_aliases(req: SuggestRequest) -> SuggestResponse:
    mgr = _get_mgr()
    default_min_sim = float(
        getattr(mgr, "suggest_min_sim", req.min_sim if req.min_sim is not None else 0.0)
    )
    candidates = mgr.suggest_aliases(
        min_sim=req.min_sim,
        limit=req.limit,
        require_inactive=req.require_inactive,
        now_ts=time.time(),
    )
    return SuggestResponse(
        candidates=[SuggestCandidate(**candidate) for candidate in candidates],
        default_min_sim=default_min_sim,
    )


@app.post(
    "/api/v1/reid/aliases/clear",
    response_model_exclude_none=True,
)
def clear_aliases() -> Dict[str, Any]:
    mgr = _get_mgr()
    cleared = mgr.clear_aliases()
    aliases = _list_aliases(mgr)
    return {"cleared": int(cleared), "aliases": aliases}


@app.get(
    "/api/v1/reid/aliases/history",
    response_model_exclude_none=True,
)
def alias_history(limit: int = Query(100, ge=1, le=5000)) -> Dict[str, Any]:
    mgr = _get_mgr()
    history = getattr(mgr, "alias_history", [])
    if not isinstance(history, list):
        history = []
    if limit > 0:
        history = history[-limit:]
    history = list(reversed(history))
    return {"history": history}
