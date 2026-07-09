from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from noesis.server.boundary_metrics import record_rest_response

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


class ResidentRecordResponse(BaseModel):
    uuid: str
    stable_id: int
    display_name: str
    created_ts: float
    embedding_count: int = 0
    gallery_embeddings: int = 0


class ResidentListResponse(BaseModel):
    residents: List[ResidentRecordResponse]
    count: int


class ResidentEnrollRequest(BaseModel):
    stable_id: Optional[int] = Field(None, ge=1)
    visitor_id: Optional[int] = Field(None, ge=1)
    display_name: str = Field(..., min_length=1)


class ResidentEnrollResponse(BaseModel):
    resident: ResidentRecordResponse
    applied: bool = True


class ResidentPatchRequest(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1)


class IdentityHealthResponse(BaseModel):
    household_mode: bool
    resident_count: int = 0
    visitor_count: int = 0
    provisional_count: int = 0
    mint_visitor_count: int = 0
    promote_resident_count: int = 0
    false_share_blocked_count: int = 0
    overlap_permit_grant_count: int = 0
    overlap_permit_deny_count: int = 0
    gallery_ids: int = 0
    gallery_quality_reject_count: int = 0
    mnn_reject_count: int = 0
    active_unique: int = 0
    residents: List[ResidentRecordResponse] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


def _resident_response(row: Dict[str, Any]) -> ResidentRecordResponse:
    return ResidentRecordResponse(
        uuid=str(row.get("uuid", "")),
        stable_id=int(row.get("stable_id", 0)),
        display_name=str(row.get("display_name", "")),
        created_ts=float(row.get("created_ts", 0.0)),
        embedding_count=int(row.get("embedding_count", 0)),
        gallery_embeddings=int(row.get("gallery_embeddings", row.get("embedding_count", 0))),
    )


@app.get(
    "/api/v1/reid/residents",
    response_model=ResidentListResponse,
    response_model_exclude_none=True,
)
def list_residents() -> ResidentListResponse:
    mgr = _get_mgr()
    if not bool(getattr(mgr, "household_mode", False)):
        raise HTTPException(status_code=400, detail="household mode not enabled")
    rows = mgr.list_residents() if hasattr(mgr, "list_residents") else []
    residents = [_resident_response(row if isinstance(row, dict) else {}) for row in rows]
    response = ResidentListResponse(residents=residents, count=len(residents))
    record_rest_response(
        "/api/v1/reid/residents:get",
        "ResidentListResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


@app.post(
    "/api/v1/reid/residents/enroll",
    response_model=ResidentEnrollResponse,
    response_model_exclude_none=True,
)
def enroll_resident(req: ResidentEnrollRequest) -> ResidentEnrollResponse:
    mgr = _get_mgr()
    if not bool(getattr(mgr, "household_mode", False)):
        raise HTTPException(status_code=400, detail="household mode not enabled")
    if not hasattr(mgr, "enroll_resident"):
        raise HTTPException(status_code=503, detail="enroll not supported")
    try:
        payload = mgr.enroll_resident(
            display_name=req.display_name,
            stable_id=req.stable_id,
            visitor_id=req.visitor_id,
            now_ts=time.time(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    resident = _resident_response(payload if isinstance(payload, dict) else {})
    response = ResidentEnrollResponse(resident=resident, applied=True)
    record_rest_response(
        "/api/v1/reid/residents/enroll:post",
        "ResidentEnrollResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


@app.patch(
    "/api/v1/reid/residents/{resident_uuid}",
    response_model=ResidentEnrollResponse,
    response_model_exclude_none=True,
)
def patch_resident(resident_uuid: str, req: ResidentPatchRequest) -> ResidentEnrollResponse:
    mgr = _get_mgr()
    if not bool(getattr(mgr, "household_mode", False)):
        raise HTTPException(status_code=400, detail="household mode not enabled")
    if not hasattr(mgr, "patch_resident"):
        raise HTTPException(status_code=503, detail="patch not supported")
    try:
        payload = mgr.patch_resident(resident_uuid, display_name=req.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    resident = _resident_response(payload if isinstance(payload, dict) else {})
    response = ResidentEnrollResponse(resident=resident, applied=True)
    record_rest_response(
        "/api/v1/reid/residents:patch",
        "ResidentEnrollResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


@app.delete(
    "/api/v1/reid/residents/{resident_uuid}",
    response_model=ResidentEnrollResponse,
    response_model_exclude_none=True,
)
def delete_resident(resident_uuid: str) -> ResidentEnrollResponse:
    mgr = _get_mgr()
    if not bool(getattr(mgr, "household_mode", False)):
        raise HTTPException(status_code=400, detail="household mode not enabled")
    if not hasattr(mgr, "delete_resident"):
        raise HTTPException(status_code=503, detail="delete not supported")
    try:
        payload = mgr.delete_resident(resident_uuid)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    resident = _resident_response(payload if isinstance(payload, dict) else {})
    response = ResidentEnrollResponse(resident=resident, applied=True)
    record_rest_response(
        "/api/v1/reid/residents:delete",
        "ResidentEnrollResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


@app.get(
    "/api/v1/reid/identity_health",
    response_model=IdentityHealthResponse,
    response_model_exclude_none=True,
)
def identity_health() -> IdentityHealthResponse:
    mgr = _get_mgr()
    if not hasattr(mgr, "get_identity_health"):
        raise HTTPException(status_code=503, detail="identity health not supported")
    try:
        payload = mgr.get_identity_health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="invalid identity health payload")
    residents = [
        _resident_response(row if isinstance(row, dict) else {})
        for row in (payload.get("residents") or [])
    ]
    response = IdentityHealthResponse(
        household_mode=bool(payload.get("household_mode")),
        resident_count=int(payload.get("resident_count", 0) or 0),
        visitor_count=int(payload.get("visitor_count", 0) or 0),
        provisional_count=int(payload.get("provisional_count", 0) or 0),
        mint_visitor_count=int(payload.get("mint_visitor_count", 0) or 0),
        promote_resident_count=int(payload.get("promote_resident_count", 0) or 0),
        false_share_blocked_count=int(payload.get("false_share_blocked_count", 0) or 0),
        overlap_permit_grant_count=int(payload.get("overlap_permit_grant_count", 0) or 0),
        overlap_permit_deny_count=int(payload.get("overlap_permit_deny_count", 0) or 0),
        gallery_ids=int(payload.get("gallery_ids", 0) or 0),
        gallery_quality_reject_count=int(payload.get("gallery_quality_reject_count", 0) or 0),
        mnn_reject_count=int(payload.get("mnn_reject_count", 0) or 0),
        active_unique=int(payload.get("active_unique", 0) or 0),
        residents=residents,
        metrics=dict(payload.get("metrics") or {}),
    )
    record_rest_response(
        "/api/v1/reid/identity_health:get",
        "IdentityHealthResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


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
    model_start_ns = time.perf_counter_ns()
    response = AliasListResponse(
        enabled=bool(getattr(mgr, "aliases_enabled", False)),
        aliases=aliases,
        alias_file=str(alias_file) if alias_file else None,
        copresence_window_s=float(getattr(mgr, "copresence_window_s", 0.0)),
        history_count=history_count,
    )
    model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
    record_rest_response(
        "/api/v1/reid/aliases:get",
        "AliasListResponse",
        model_duration_ms=model_ms,
        payload=response,
        include_budget=True,
    )
    return response


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
    model_start_ns = time.perf_counter_ns()
    response = _merge_payload_to_response(payload, fallback_a=req.a, fallback_b=req.b, aliases=aliases)
    model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
    record_rest_response(
        "/api/v1/reid/aliases/merge:post",
        "MergeResponse",
        model_duration_ms=model_ms,
        payload=response,
        include_budget=True,
    )
    return response


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

    model_start_ns = time.perf_counter_ns()
    response = MergeBatchResponse(
        results=results,
        applied_count=applied_count,
        failed_count=failed_count,
        aliases=aliases,
    )
    model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
    record_rest_response(
        "/api/v1/reid/aliases/merge-batch:post",
        "MergeBatchResponse",
        model_duration_ms=model_ms,
        payload=response,
        include_budget=True,
    )
    return response


@app.post(
    "/api/v1/reid/aliases/unset",
    response_model=UnsetResponse,
    response_model_exclude_none=True,
)
def unset_alias(req: UnsetRequest) -> UnsetResponse:
    mgr = _get_mgr()
    payload = mgr.unset_alias(req.src)
    aliases = _list_aliases(mgr)
    model_start_ns = time.perf_counter_ns()
    response = UnsetResponse(
        src=int(payload.get("src", req.src)),
        removed=bool(payload.get("removed", False)),
        reason=payload.get("reason"),
        aliases=aliases,
    )
    model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
    record_rest_response(
        "/api/v1/reid/aliases/unset:post",
        "UnsetResponse",
        model_duration_ms=model_ms,
        payload=response,
        include_budget=True,
    )
    return response


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
    model_start_ns = time.perf_counter_ns()
    response = SuggestResponse(
        candidates=[SuggestCandidate(**candidate) for candidate in candidates],
        default_min_sim=default_min_sim,
    )
    model_ms = (time.perf_counter_ns() - model_start_ns) / 1_000_000.0
    record_rest_response(
        "/api/v1/reid/aliases/suggest:post",
        "SuggestResponse",
        model_duration_ms=model_ms,
        payload=response,
        include_budget=True,
    )
    return response


@app.post(
    "/api/v1/reid/aliases/clear",
    response_model_exclude_none=True,
)
def clear_aliases() -> Dict[str, Any]:
    mgr = _get_mgr()
    cleared = mgr.clear_aliases()
    aliases = _list_aliases(mgr)
    response = {"cleared": int(cleared), "aliases": aliases}
    record_rest_response(
        "/api/v1/reid/aliases/clear:post",
        "DictResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response


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
    response = {"history": history}
    record_rest_response(
        "/api/v1/reid/aliases/history:get",
        "DictResponse",
        model_duration_ms=0.0,
        payload=response,
        include_budget=True,
    )
    return response
