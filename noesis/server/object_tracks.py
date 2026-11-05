"""Fixed internal-bearer object continuity and actor-owned selection routes."""
from __future__ import annotations
import asyncio,json,sqlite3,threading
from fastapi import APIRouter,HTTPException,Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from noesis_core.contracts.object_tracks import ObjectSelectionCommand,ObjectSelectionConfirm,ObjectSelectionsRead,MAX_BYTES
from noesis_core.object_tracks import ObjectTrackService,ObjectTrackConflict,ObjectTrackMissing,ObjectTrackError,canonical

PATH='/api/v1/objects/tracks'
SELECTIONS='/api/v1/objects/selections'
router=APIRouter()
_workers=threading.BoundedSemaphore(2)

def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('duplicate field')
        result[key]=value
    return result

def _perform(request,action,command):
    owner=getattr(request.app.state,'object_track_service',None)
    if type(owner) is not ObjectTrackService:raise HTTPException(503,detail='object_tracks_unbound')
    try:
        result=owner.snapshot() if action=='snapshot' else owner.read(command.actor_id) if action=='read' else getattr(owner,action)(command)
        if action=='prepare':
            remaining=(result.expires_at_us-owner._now())//1000
            if remaining<=0:raise ObjectTrackConflict('object_track_review_expired')
            result=result.model_copy(update={'remaining_ms':remaining})
        body=canonical(result).encode()
        if getattr(request.app.state,'object_track_service',None) is not owner or owner._closed:raise HTTPException(503,detail='object_tracks_owner_retired')
        return Response(body,media_type='application/json',headers={'Cache-Control':'private, no-store'})
    except ObjectTrackMissing as exc:raise HTTPException(404,detail=str(exc)) from exc
    except ObjectTrackConflict as exc:raise HTTPException(409,detail=str(exc)) from exc
    except (ObjectTrackError,sqlite3.Error,ValueError) as exc:raise HTTPException(503,detail='object_tracks_unavailable') from exc

async def _handle(request,action,model=None):
    if request.scope.get('query_string') or request.headers.get('transfer-encoding'):raise HTTPException(422,detail='object_tracks_query_not_allowed')
    if model is not None and request.headers.get('content-type','').split(';',1)[0].strip().lower()!='application/json':raise HTTPException(415,detail='object_tracks_json_required')
    data=bytearray()
    try:
        async with asyncio.timeout(2):
            async for chunk in request.stream():
                if len(data)+len(chunk)>MAX_BYTES:raise HTTPException(413,detail='object_tracks_request_too_large')
                data.extend(chunk)
        if model is None:
            if data:raise ValueError('GET body')
            command=None
        else:
            command=model.model_validate(json.loads(data.decode(),object_pairs_hook=_pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite'))))
    except (ValueError,TypeError,RecursionError):raise HTTPException(422,detail='object_tracks_request_invalid')
    except TimeoutError:raise HTTPException(408,detail='object_tracks_request_timeout')
    if not _workers.acquire(blocking=False):raise HTTPException(503,detail='object_tracks_busy')
    def work():
        try:return _perform(request,action,command)
        finally:_workers.release()
    return await run_in_threadpool(work)

@router.get(PATH)
async def tracks(request:Request):return await _handle(request,'snapshot')
@router.post(SELECTIONS+'/read')
async def selections(request:Request):return await _handle(request,'read',ObjectSelectionsRead)
@router.post(SELECTIONS+'/prepare')
async def prepare(request:Request):return await _handle(request,'prepare',ObjectSelectionCommand)
@router.put(SELECTIONS)
async def confirm(request:Request):return await _handle(request,'confirm',ObjectSelectionConfirm)
@router.post(SELECTIONS+'/lookup')
async def lookup(request:Request):return await _handle(request,'lookup',ObjectSelectionCommand)
