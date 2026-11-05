"""Private manually requested domain map using the existing bounded source owner."""
from fastapi import APIRouter, Path, Request
from noesis_core.contracts.room_coverage import RoomCoverageRequest
from .view_coverage import _handle

router = APIRouter()

@router.post('/api/v1/scene-priors/cameras/{camera_id}/domain-coverage')
async def domain_coverage(request: Request, camera_id: str = Path(..., min_length=1, max_length=160, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')):
    return await _handle(request, camera_id, False, RoomCoverageRequest)
