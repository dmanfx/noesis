"""Source-scoped native tracker continuity, not physical/global identity."""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator
from .object_associations import AssociationModel, source_text, clean_text
from .object_observations import ObjectImageBox
from .semantic_space import StableId
from .base import Sha256

MAX_BYTES=262144
MAX_TRACKS=512
MAX_SELECTIONS=64
MAX_OPERATIONS=128
REPLAY_US=86400_000_000
Integer=Annotated[StrictInt,Field(ge=0,le=2**53-1)]
Timestamp=Annotated[StrictInt,Field(ge=1,le=2**53-1)]
Confidence=Annotated[float,Field(ge=0,le=1,allow_inf_nan=False)]

class NativeObjectKey(AssociationModel):
    run_id: str=Field(min_length=1,max_length=160)
    camera_id: str=Field(min_length=1,max_length=160)
    source_id: Integer
    source_epoch: Integer
    continuity_generation: Integer
    native_id: str=Field(pattern=r'^(0|[1-9][0-9]{0,19})$')
    _text=field_validator('run_id','camera_id')(source_text)
    @field_validator('native_id')
    @classmethod
    def native(cls,value):
        if int(value)>=2**64-1: raise ValueError('untracked sentinel is not an identity')
        return value

SpatialReason=Literal['nonperson_spatial_measurement_not_admitted','calibration_unavailable','original_depth_unavailable','projection_undeclared_or_unsupported','depth_sample_invalid','calibration_replaced','measured_surface','sampling_budget','sampling_failed']
class ObjectSpatial(AssociationModel):
    frame_id: Integer
    observed_at_us: Timestamp
    media_pts_ns: Integer
    quantity: Literal['image_center_surface']='image_center_surface'
    method: Literal['metric_monocular_depth']='metric_monocular_depth'
    confidence: Literal['estimated']='estimated'
    world_frame: Literal['backend_world_m']='backend_world_m'
    world_frame_revision: str=Field(min_length=1,max_length=256)
    frame_transform_sha256: Sha256
    camera_calibration_sha256: Sha256
    world_alignment_sha256: Sha256
    position_m: tuple[Annotated[float,Field(ge=-1000,le=1000,allow_inf_nan=False)],Annotated[float,Field(ge=-1000,le=1000,allow_inf_nan=False)],Annotated[float,Field(ge=-1000,le=1000,allow_inf_nan=False)]]
    pixel: tuple[Integer,Integer]
    depth_m: Annotated[float,Field(ge=.05,le=100,allow_inf_nan=False)]
    model: str=Field(min_length=1,max_length=160)

class ObjectTrackSample(AssociationModel):
    observation_id: str=Field(min_length=1,max_length=1400)
    frame_id: Integer
    observed_at_us: Timestamp
    captured_at_us: Timestamp
    capture_time_status: Literal['estimated']='estimated'
    observation_clock: Literal['processing_unix_epoch_us']='processing_unix_epoch_us'
    media_pts_ns: Integer
    media_pts_domain: Literal['stream_relative_ns']='stream_relative_ns'
    coordinate_frame: Literal['pipeline_image_px']='pipeline_image_px'
    image_width: Annotated[StrictInt,Field(ge=1,le=32768)]
    image_height: Annotated[StrictInt,Field(ge=1,le=32768)]
    source_image_width: Annotated[StrictInt,Field(ge=1,le=32768)]
    source_image_height: Annotated[StrictInt,Field(ge=1,le=32768)]
    class_id: Annotated[StrictInt,Field(ge=1,le=79)]
    label: str=Field(min_length=1,max_length=160)
    detector_confidence: Confidence|None
    tracker_confidence: Confidence|None
    evidence: Literal['detector_and_tracker','tracker_only']
    bbox: ObjectImageBox
    room_id: str|None=Field(default=None,min_length=1,max_length=160)
    room_source: Literal['nvdsanalytics_roi']|None=None
    spatial: ObjectSpatial|None=None
    spatial_reason: SpatialReason='nonperson_spatial_measurement_not_admitted'
    _room=field_validator('room_id')(lambda v: source_text(v) if v is not None else v)
    _text=field_validator('observation_id','label')(source_text)
    @model_validator(mode='after')
    def bounds(self):
        if self.bbox.left+self.bbox.width>self.image_width or self.bbox.top+self.bbox.height>self.image_height: raise ValueError('image bounds')
        if (self.room_id is None)!=(self.room_source is None):raise ValueError('room source required')
        if (self.spatial is not None)!=(self.spatial_reason=='measured_surface'):raise ValueError('spatial reason mismatch')
        if self.spatial and (self.spatial.pixel[0]>=self.image_width or self.spatial.pixel[1]>=self.image_height):raise ValueError('surface pixel bounds')
        return self

Lifecycle=Literal['observed','not_observed','source_lost','continuity_gap','source_replaced','run_replaced','runtime_unavailable','terminated']
class ObjectTrack(AssociationModel):
    key: NativeObjectKey
    last_seen: ObjectTrackSample
    state: Lifecycle
    spatial: ObjectSpatial|None=None
    spatial_reason: SpatialReason='nonperson_spatial_measurement_not_admitted'
    termination_reason: Literal['native_nonperson_termination_not_exposed','native_termination_observed','native_termination_available']='native_nonperson_termination_not_exposed'

class ObjectTracksSnapshot(AssociationModel):
    contract: Literal['noesis.object-tracks']='noesis.object-tracks'
    version: Literal[1]=1
    run_id: str|None
    status: Literal['available','partial','unknown']
    reason: Literal['native_tracker_evidence','no_input','closed','queue_dropped','source_lost','native_id_unavailable']
    generated_at_us: Timestamp
    dropped_frames: Integer
    untracked_count: Integer
    omitted_count: Integer
    tracks: Annotated[tuple[ObjectTrack,...],Field(max_length=MAX_TRACKS)]

class SelectionInput(AssociationModel):
    id: StableId
    label: str=Field(min_length=1,max_length=80)
    retired: StrictBool
    observation_id: str|None=Field(min_length=1,max_length=1400)
    _label=field_validator('label')(clean_text)
    _source=field_validator('observation_id')(lambda v: source_text(v) if v is not None else v)
class ObjectSelectionCommand(AssociationModel):
    actor_id: str=Field(min_length=1,max_length=160)
    operation_id: StableId
    expected_revision: StableId|None
    selection: SelectionInput
    _actor=field_validator('actor_id')(clean_text)
class ObjectSelection(AssociationModel):
    id: StableId
    label: str=Field(min_length=1,max_length=80)
    retired: StrictBool
    key: NativeObjectKey
    original: ObjectTrackSample
    last_seen: ObjectTrackSample
    state: Lifecycle
    spatial: ObjectSpatial|None=None
    spatial_reason: SpatialReason='nonperson_spatial_measurement_not_admitted'
    termination_reason: Literal['native_nonperson_termination_not_exposed','native_termination_observed','native_termination_available']='native_nonperson_termination_not_exposed'
class ObjectSelectionRecord(AssociationModel):
    revision: StableId|None
    updated_at_us: Timestamp|None
    selections: Annotated[tuple[ObjectSelection,...],Field(max_length=MAX_SELECTIONS)]
class ObjectSelectionsRead(AssociationModel):
    actor_id: str=Field(min_length=1,max_length=160)
    _actor=field_validator('actor_id')(clean_text)
class ObjectSelectionsSnapshot(AssociationModel):
    contract: Literal['noesis.object-selections']='noesis.object-selections'
    version: Literal[1]=1
    record: ObjectSelectionRecord
class ObjectSelectionResult(AssociationModel):
    contract: Literal['noesis.object-selections.result']='noesis.object-selections.result'
    version: Literal[1]=1
    operation_id: StableId
    record: ObjectSelectionRecord
    replayed: StrictBool

class ObjectSelectionReview(AssociationModel):
    contract: Literal['noesis.object-selections.review']='noesis.object-selections.review'
    version: Literal[1]=1
    review_token: str=Field(pattern=r'^[A-Za-z0-9_-]{43}$')
    command: ObjectSelectionCommand
    selection: ObjectSelection
    expires_at_us: Timestamp
    remaining_ms: Annotated[StrictInt,Field(ge=1,le=120000)]
class ObjectSelectionConfirm(AssociationModel):
    review_token: str=Field(pattern=r'^[A-Za-z0-9_-]{43}$')
    command: ObjectSelectionCommand
