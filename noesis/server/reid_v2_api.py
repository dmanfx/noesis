"""Typed, opt-in FastAPI surface for the store-backed identity-v2 runtime."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from noesis.server.boundary_metrics import (
    BoundaryMetricsRoute,
    mark_rest_response,
    measure_rest_response_model,
)

from reid.identity_v2 import (
    DuplicateResidentError,
    EnrollmentObservationKey,
    EnrollmentProposalConflict,
    EnrollmentProposalExpired,
    EnrollmentProposalRecord,
    IdentityHealth,
    IdentityStoreError,
    IdentityV2Runtime,
    ModelProfileMismatch,
    ObservationEvidenceConflict,
    ObservationEvidenceConsumed,
    ObservationEvidenceExpired,
    ObservationEvidenceUnavailable,
    ResidentRecord,
)

router = APIRouter(
    prefix="/api/v2/reid",
    tags=["reid-v2"],
    route_class=BoundaryMetricsRoute,
)

_RUNTIME_GETTER: Optional[Callable[[], Optional[IdentityV2Runtime]]] = None
_SERVICE_GETTER: Optional[Callable[[], Optional[Any]]] = None


def register_identity_v2_runtime_getter(
    getter: Callable[[], Optional[IdentityV2Runtime]],
) -> None:
    global _RUNTIME_GETTER
    if not callable(getter):
        raise TypeError("identity-v2 runtime getter must be callable")
    _RUNTIME_GETTER = getter


def register_identity_v2_service_getter(
    getter: Callable[[], Optional[Any]],
) -> None:
    global _SERVICE_GETTER
    if not callable(getter):
        raise TypeError("identity-v2 service getter must be callable")
    _SERVICE_GETTER = getter


def clear_identity_v2_bindings() -> None:
    """Release process-local identity owners after shutdown or startup abort."""
    global _RUNTIME_GETTER, _SERVICE_GETTER
    _RUNTIME_GETTER = None
    _SERVICE_GETTER = None


def get_identity_v2_runtime() -> IdentityV2Runtime:
    if _RUNTIME_GETTER is None:
        raise HTTPException(
            status_code=503, detail="identity-v2 runtime not registered"
        )
    runtime = _RUNTIME_GETTER()
    if runtime is None:
        raise HTTPException(status_code=503, detail="identity-v2 runtime unavailable")
    service = _SERVICE_GETTER() if _SERVICE_GETTER is not None else None
    if service is not None and bool(getattr(service, "closed", False)):
        raise HTTPException(status_code=503, detail="identity-v2 runtime is closed")
    return runtime


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationKeyModel(StrictRequestModel):
    run_id: str = Field(..., min_length=1)
    camera_id: str = Field(..., min_length=1)
    tracker_id: str = Field(..., min_length=1)
    frame_id: int = Field(..., ge=0)
    observation_id: str = Field(..., min_length=1)

    def to_domain(self) -> EnrollmentObservationKey:
        return EnrollmentObservationKey(**self.model_dump())


class ResidentResponse(BaseModel):
    resident_uuid: str
    display_name: str
    normalized_name: str
    compatibility_sid: int
    model_fingerprint: str
    embedding_dim: int
    created_at: float
    updated_at: float


class ResidentListResponse(BaseModel):
    residents: List[ResidentResponse]
    count: int


class ModelProfileResponse(BaseModel):
    model_fingerprint: str
    embedding_dim: int
    resident_count: int


class IdentityV2HealthResponse(BaseModel):
    schema_version: int
    resident_count: int
    residents_with_anchors: int
    residents_without_anchors: int
    enrollment_anchor_count: int
    adaptive_exemplar_count: int
    quarantine_exemplar_count: int
    active_visitor_sessions: int
    visitor_slot_count: int
    active_provisional_sessions: int
    visitor_exemplar_count: int
    pending_enrollment_proposals: int
    confirmed_enrollment_proposals: int
    model_profiles: List[ModelProfileResponse]
    scoring_calibration_status: str
    scoring_calibration_artifact_id: Optional[str]
    scoring_authority_scope: Optional[str]
    public_authority_cutover_status: Optional[str]
    authority_cutover_artifact_id: Optional[str]
    authority_runtime_profile_sha256: Optional[str]
    runtime_mode: Optional[str] = None
    runtime_model_fingerprint: str
    runtime_model_semantic_profile_sha256: Optional[str]
    runtime_model_layer: Optional[str] = None
    runtime_embedding_dim: int
    calibrated_maximum_resident_candidates: Optional[int]
    calibrated_maximum_visitor_candidates: Optional[int]
    calibrated_maximum_total_candidates: Optional[int]
    calibrated_maximum_exemplars_per_candidate: Optional[int]
    observation_cache_entries: int
    observation_cache_max_entries: int
    observation_cache_ttl_s: float


class IdentityEvidenceStatusResponse(BaseModel):
    enabled: bool
    contract: str
    contract_version: int
    session_id: Optional[str] = None
    source: Optional[str] = None
    runtime: Optional[str] = None
    recorded_event_count: int = 0
    last_observed_at_us: Optional[int] = None
    retained_bytes: int = 0
    max_records: Optional[int] = None
    max_bytes: Optional[int] = None
    max_age_s: Optional[float] = None
    pruned_event_count: int = 0
    contains_embeddings: bool = False


class MigrationReviewResponse(BaseModel):
    available: bool
    report: Optional[Dict[str, Any]] = None
    apply_available_from_api: bool = False


class EnrollmentProposalRequest(StrictRequestModel):
    key: ObservationKeyModel
    display_name: str = Field(..., min_length=1)
    compatibility_sid: Optional[int] = Field(None, ge=1)
    update_resident_uuid: Optional[str] = None
    ttl_s: float = Field(300.0, gt=0.0, le=3600.0)


class EnrollmentConfirmationRequest(StrictRequestModel):
    key: ObservationKeyModel
    evidence_digest: str = Field(..., min_length=64, max_length=64)


class EnrollmentProposalResponse(BaseModel):
    proposal_uuid: str
    key: ObservationKeyModel
    observation_quality: float
    observation_evidence: List[str]
    display_name: str
    normalized_name: str
    compatibility_sid: Optional[int]
    target_resident_uuid: Optional[str]
    proposed_resident_uuid: Optional[str]
    action: str
    model_fingerprint: str
    embedding_dim: int
    evidence_digest: str
    created_at: float
    expires_at: float
    state: str
    confirmed_at: Optional[float]
    result_resident_uuid: Optional[str]
    anchor_exemplar_uuid: Optional[str]
    idempotent: bool = False


class EnrollmentProposalListResponse(BaseModel):
    proposals: List[EnrollmentProposalResponse]
    count: int


class EnrollmentConfirmationResponse(BaseModel):
    proposal: EnrollmentProposalResponse
    resident: ResidentResponse
    anchor_exemplar_uuid: str
    idempotent: bool


class ResidentPatchRequest(StrictRequestModel):
    display_name: str = Field(..., min_length=1)


class ResidentMutationResponse(BaseModel):
    resident: ResidentResponse
    applied: bool = True


class ResidentDeletionResponse(BaseModel):
    resident_uuid: str
    resident_deleted: bool
    exemplars_deleted: int


def _resident_response(row: ResidentRecord) -> ResidentResponse:
    return ResidentResponse(
        resident_uuid=row.resident_uuid,
        display_name=row.display_name,
        normalized_name=row.normalized_name,
        compatibility_sid=row.compatibility_sid,
        model_fingerprint=row.model_fingerprint,
        embedding_dim=row.embedding_dim,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _health_response(
    row: IdentityHealth,
    runtime: IdentityV2Runtime,
) -> IdentityV2HealthResponse:
    service = _SERVICE_GETTER() if _SERVICE_GETTER is not None else None
    cache = runtime.observation_cache_health()
    policy = runtime.resolver.scorer.policy
    return IdentityV2HealthResponse(
        schema_version=row.schema_version,
        resident_count=row.resident_count,
        residents_with_anchors=row.residents_with_anchors,
        residents_without_anchors=row.residents_without_anchors,
        enrollment_anchor_count=row.enrollment_anchor_count,
        adaptive_exemplar_count=row.adaptive_exemplar_count,
        quarantine_exemplar_count=row.quarantine_exemplar_count,
        active_visitor_sessions=row.active_visitor_sessions,
        visitor_slot_count=row.visitor_slot_count,
        active_provisional_sessions=row.active_provisional_sessions,
        visitor_exemplar_count=row.visitor_exemplar_count,
        pending_enrollment_proposals=row.pending_enrollment_proposals,
        confirmed_enrollment_proposals=row.confirmed_enrollment_proposals,
        model_profiles=[
            ModelProfileResponse(
                model_fingerprint=fingerprint,
                embedding_dim=dimension,
                resident_count=count,
            )
            for fingerprint, dimension, count in row.model_profiles
        ],
        scoring_calibration_status=runtime.scoring_calibration_status,
        scoring_calibration_artifact_id=runtime.scoring_calibration_artifact_id,
        scoring_authority_scope=(
            "open_set_scorer_policy_only"
            if runtime.scoring_calibration_status == "artifact_backed"
            else None
        ),
        public_authority_cutover_status=(
            "verified"
            if bool(getattr(service, "authoritative", False))
            and bool(getattr(service, "authority_cutover_artifact_id", None))
            else "blocked" if service is not None else None
        ),
        authority_cutover_artifact_id=(
            str(getattr(service, "authority_cutover_artifact_id", "") or "") or None
        ),
        authority_runtime_profile_sha256=(
            str(getattr(service, "authority_runtime_profile_sha256", "") or "") or None
        ),
        runtime_mode=(
            str(getattr(getattr(service, "mode", None), "value", "") or "") or None
        ),
        runtime_model_fingerprint=runtime.model_fingerprint,
        runtime_model_semantic_profile_sha256=(policy.model_semantic_profile_sha256),
        runtime_model_layer=(str(getattr(service, "model_layer", "") or "") or None),
        runtime_embedding_dim=runtime.embedding_dim,
        calibrated_maximum_resident_candidates=(policy.maximum_resident_candidates),
        calibrated_maximum_visitor_candidates=(policy.maximum_visitor_candidates),
        calibrated_maximum_total_candidates=policy.maximum_total_candidates,
        calibrated_maximum_exemplars_per_candidate=(
            policy.maximum_exemplars_per_candidate
        ),
        observation_cache_entries=cache.entry_count,
        observation_cache_max_entries=cache.max_entries,
        observation_cache_ttl_s=cache.ttl_s,
    )


def _proposal_response(
    row: EnrollmentProposalRecord,
    *,
    idempotent: bool = False,
    now: Optional[float] = None,
) -> EnrollmentProposalResponse:
    return EnrollmentProposalResponse(
        proposal_uuid=row.proposal_uuid,
        key=ObservationKeyModel(
            run_id=row.key.run_id,
            camera_id=row.key.camera_id,
            tracker_id=row.key.tracker_id,
            frame_id=row.key.frame_id,
            observation_id=row.key.observation_id,
        ),
        observation_quality=row.observation_quality,
        observation_evidence=list(row.observation_evidence),
        display_name=row.display_name,
        normalized_name=row.normalized_name,
        compatibility_sid=row.compatibility_sid,
        target_resident_uuid=row.target_resident_uuid,
        proposed_resident_uuid=row.proposed_resident_uuid,
        action=row.action,
        model_fingerprint=row.model_fingerprint,
        embedding_dim=row.embedding_dim,
        evidence_digest=row.evidence_digest,
        created_at=row.created_at,
        expires_at=row.expires_at,
        state=row.effective_state(now=now),
        confirmed_at=row.confirmed_at,
        result_resident_uuid=row.result_resident_uuid,
        anchor_exemplar_uuid=row.anchor_exemplar_uuid,
        idempotent=idempotent,
    )


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
    if isinstance(exc, EnrollmentProposalExpired):
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if isinstance(exc, ObservationEvidenceExpired):
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if isinstance(
        exc,
        (
            ObservationEvidenceUnavailable,
            ObservationEvidenceConsumed,
            ObservationEvidenceConflict,
        ),
    ):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (DuplicateResidentError, EnrollmentProposalConflict)):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ModelProfileMismatch):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, IdentityStoreError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    raise exc


@router.get("/health", response_model=IdentityV2HealthResponse)
def identity_v2_health(request: Request) -> IdentityV2HealthResponse:
    runtime = get_identity_v2_runtime()
    health = runtime.health()
    with measure_rest_response_model(
        "/api/v2/reid/health:get", "IdentityV2HealthResponse"
    ) as model_measurement:
        response = _health_response(health, runtime)
    mark_rest_response(
        request,
        "/api/v2/reid/health:get",
        "IdentityV2HealthResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/evidence/status", response_model=IdentityEvidenceStatusResponse)
def identity_v2_evidence_status(request: Request) -> IdentityEvidenceStatusResponse:
    get_identity_v2_runtime()
    service = _SERVICE_GETTER() if _SERVICE_GETTER is not None else None
    recorder = getattr(service, "evidence_recorder", None)
    if recorder is None:
        with measure_rest_response_model(
            "/api/v2/reid/evidence/status:get", "IdentityEvidenceStatusResponse"
        ) as model_measurement:
            response = IdentityEvidenceStatusResponse(
                enabled=False,
                contract="noesis.identity.shadow_score_evidence",
                contract_version=2,
            )
    else:
        row = recorder.health()
        with measure_rest_response_model(
            "/api/v2/reid/evidence/status:get", "IdentityEvidenceStatusResponse"
        ) as model_measurement:
            response = IdentityEvidenceStatusResponse(
                enabled=True,
                contract=row.contract,
                contract_version=row.contract_version,
                session_id=row.session_id,
                source=row.source,
                runtime=row.runtime,
                recorded_event_count=row.recorded_event_count,
                last_observed_at_us=row.last_observed_at_us,
                retained_bytes=int(getattr(row, "retained_bytes", 0)),
                max_records=getattr(row, "max_records", None),
                max_bytes=getattr(row, "max_bytes", None),
                max_age_s=getattr(row, "max_age_s", None),
                pruned_event_count=int(getattr(row, "pruned_event_count", 0)),
            )
    mark_rest_response(
        request,
        "/api/v2/reid/evidence/status:get",
        "IdentityEvidenceStatusResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/migration/review", response_model=MigrationReviewResponse)
def identity_v2_migration_review(request: Request) -> MigrationReviewResponse:
    get_identity_v2_runtime()
    service = _SERVICE_GETTER() if _SERVICE_GETTER is not None else None
    review = getattr(service, "migration_review", None)
    with measure_rest_response_model(
        "/api/v2/reid/migration/review:get", "MigrationReviewResponse"
    ) as model_measurement:
        response = (
            MigrationReviewResponse(available=False)
            if not isinstance(review, dict)
            else MigrationReviewResponse(available=True, report=dict(review))
        )
    mark_rest_response(
        request,
        "/api/v2/reid/migration/review:get",
        "MigrationReviewResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get("/residents", response_model=ResidentListResponse)
def list_identity_v2_residents(request: Request) -> ResidentListResponse:
    rows = get_identity_v2_runtime().list_residents()
    with measure_rest_response_model(
        "/api/v2/reid/residents:get", "ResidentListResponse"
    ) as model_measurement:
        residents = [_resident_response(row) for row in rows]
        response = ResidentListResponse(residents=residents, count=len(residents))
    mark_rest_response(
        request,
        "/api/v2/reid/residents:get",
        "ResidentListResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get(
    "/enrollment/proposals",
    response_model=EnrollmentProposalListResponse,
)
def list_enrollment_proposals(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
) -> EnrollmentProposalListResponse:
    runtime = get_identity_v2_runtime()
    now = runtime.current_time()
    proposal_rows = runtime.list_enrollment_proposals(limit=limit)
    with measure_rest_response_model(
        "/api/v2/reid/enrollment/proposals:get", "EnrollmentProposalListResponse"
    ) as model_measurement:
        rows = [_proposal_response(row, now=now) for row in proposal_rows]
        response = EnrollmentProposalListResponse(proposals=rows, count=len(rows))
    mark_rest_response(
        request,
        "/api/v2/reid/enrollment/proposals:get",
        "EnrollmentProposalListResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.get(
    "/enrollment/proposals/{proposal_uuid}",
    response_model=EnrollmentProposalResponse,
)
def get_enrollment_proposal(
    proposal_uuid: str,
    request: Request,
) -> EnrollmentProposalResponse:
    try:
        runtime = get_identity_v2_runtime()
        row = runtime.get_enrollment_proposal(proposal_uuid)
        now = runtime.current_time()
    except Exception as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    with measure_rest_response_model(
        "/api/v2/reid/enrollment/proposals/{proposal_uuid}:get",
        "EnrollmentProposalResponse",
    ) as model_measurement:
        response = _proposal_response(row, now=now)
    mark_rest_response(
        request,
        "/api/v2/reid/enrollment/proposals/{proposal_uuid}:get",
        "EnrollmentProposalResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.post(
    "/enrollment/proposals",
    response_model=EnrollmentProposalResponse,
    status_code=201,
)
def propose_enrollment(
    request: EnrollmentProposalRequest,
    http_request: Request,
) -> EnrollmentProposalResponse:
    try:
        runtime = get_identity_v2_runtime()
        result = runtime.propose_enrollment_from_cache(
            request.key.to_domain(),
            display_name=request.display_name,
            compatibility_sid=request.compatibility_sid,
            update_resident_uuid=request.update_resident_uuid,
            ttl_s=request.ttl_s,
        )
        now = runtime.current_time()
    except Exception as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    with measure_rest_response_model(
        "/api/v2/reid/enrollment/proposals:post", "EnrollmentProposalResponse"
    ) as model_measurement:
        response = _proposal_response(
            result.proposal,
            idempotent=result.idempotent,
            now=now,
        )
    mark_rest_response(
        http_request,
        "/api/v2/reid/enrollment/proposals:post",
        "EnrollmentProposalResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.post(
    "/enrollment/proposals/{proposal_uuid}/confirm",
    response_model=EnrollmentConfirmationResponse,
)
def confirm_enrollment(
    proposal_uuid: str,
    request: EnrollmentConfirmationRequest,
    http_request: Request,
) -> EnrollmentConfirmationResponse:
    try:
        runtime = get_identity_v2_runtime()
        result = runtime.confirm_enrollment(
            proposal_uuid,
            expected_key=request.key.to_domain(),
            evidence_digest=request.evidence_digest,
        )
        now = runtime.current_time()
    except Exception as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    with measure_rest_response_model(
        "/api/v2/reid/enrollment/proposals/{proposal_uuid}/confirm:post",
        "EnrollmentConfirmationResponse",
    ) as model_measurement:
        response = EnrollmentConfirmationResponse(
            proposal=_proposal_response(
                result.proposal,
                idempotent=result.idempotent,
                now=now,
            ),
            resident=_resident_response(result.resident),
            anchor_exemplar_uuid=result.anchor.exemplar_uuid,
            idempotent=result.idempotent,
        )
    mark_rest_response(
        http_request,
        "/api/v2/reid/enrollment/proposals/{proposal_uuid}/confirm:post",
        "EnrollmentConfirmationResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.patch(
    "/residents/{resident_uuid}",
    response_model=ResidentMutationResponse,
)
def patch_identity_v2_resident(
    resident_uuid: str,
    request: ResidentPatchRequest,
    http_request: Request,
) -> ResidentMutationResponse:
    try:
        resident = get_identity_v2_runtime().update_resident_display_name(
            resident_uuid,
            display_name=request.display_name,
        )
    except Exception as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    with measure_rest_response_model(
        "/api/v2/reid/residents/{resident_uuid}:patch", "ResidentMutationResponse"
    ) as model_measurement:
        response = ResidentMutationResponse(resident=_resident_response(resident))
    mark_rest_response(
        http_request,
        "/api/v2/reid/residents/{resident_uuid}:patch",
        "ResidentMutationResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


@router.delete(
    "/residents/{resident_uuid}",
    response_model=ResidentDeletionResponse,
)
def delete_identity_v2_resident(
    resident_uuid: str,
    request: Request,
) -> ResidentDeletionResponse:
    try:
        result = get_identity_v2_runtime().delete_resident(resident_uuid)
        if not result.resident_deleted:
            raise KeyError(f"unknown resident UUID: {resident_uuid}")
    except Exception as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    with measure_rest_response_model(
        "/api/v2/reid/residents/{resident_uuid}:delete", "ResidentDeletionResponse"
    ) as model_measurement:
        response = ResidentDeletionResponse(
            resident_uuid=resident_uuid,
            resident_deleted=True,
            exemplars_deleted=result.exemplars_deleted,
        )
    mark_rest_response(
        request,
        "/api/v2/reid/residents/{resident_uuid}:delete",
        "ResidentDeletionResponse",
        model_duration_ms=model_measurement.elapsed_ms,
    )
    return response


__all__ = [
    "get_identity_v2_runtime",
    "register_identity_v2_runtime_getter",
    "register_identity_v2_service_getter",
    "clear_identity_v2_bindings",
    "router",
]
