export type Vec3 = readonly [number, number, number];
export type Cohort = Readonly<{
  source_id: number; frame_id: number | null; observed_at_us: number | null;
  tracking_publication_sequence: number;
}>;
export type JsonRecord = Readonly<Record<string, unknown>>;
export type UnifiedBevBindings = Readonly<{
  coordinateFrame: 'menon_scene'; authoredSceneSha256: string; registrationSha256: string;
  cameras: Readonly<Record<string, JsonRecord>>;
  presentations: Readonly<Record<string, JsonRecord>>;
}>;
export type UnifiedBevMap = Readonly<{
  cameraId: string; mapId: string; coordinateFrame: 'menon_scene'; sourceFrame: 'camera_local_ground_m';
  worldFrameRevision: string; worldTransformSha256: string; sceneRevisionId: string;
  sceneMappingSha256: string; authoredSceneSha256: string; snapshotContentSha256: string;
  staticObservedAtUs: number; positionAuthority: 'static_only';
  localBounds: Readonly<{ min_x: number; max_x: number; min_z: number; max_z: number }>;
  gridShape: readonly [number, number]; rasterOrientation: string;
  sceneFromLocalColMajor: readonly number[]; sceneCorners: readonly Vec3[];
}>;
export type UnifiedBevTrailPoint = Readonly<{
  position: Vec3; worldPosition: Vec3; observedAtUs: number; positionState: 'measured';
  cohort: Cohort; worldFrameRevision: string; sceneMappingSha256: string;
}>;
export type UnifiedBevPerson = Readonly<{
  entityId: string; subjectId: string; stableId: number | null; identityKind: string; generation: number; label: string;
  worldPosition: Vec3; position: Vec3; coordinateFrame: 'menon_scene'; sceneTransformCount: 1;
  worldCovariance: readonly number[]; sceneCovariance: readonly number[];
  worldVelocity: Vec3 | null; sceneVelocity: Vec3 | null;
  supportState: string; posture: string; supportSurface: JsonRecord | null;
  lifecycle: 'present' | 'held'; positionState: 'measured' | 'predicted' | 'held';
  admission: 'accepted' | 'predicted' | 'held'; observedAtUs: number; staleAfterUs: number; expiresAtMs: number;
  continuitySegmentId: number | null; worldFrameRevision: string; worldTransformSha256: string | null;
  calibrationRevision: string | null; sceneRevisionId: string; sceneMappingSha256: string; authoredSceneSha256: string;
  sourceEvidence: readonly JsonRecord[]; conflict: boolean; roomId: string | null;
  trailKey: string; trail: readonly UnifiedBevTrailPoint[]; trailBreak: boolean; cohort: Cohort;
}>;
export type UnifiedBevState = Readonly<{
  authoredSceneSha256: string | null; maxTrailPoints: number; bindings: UnifiedBevBindings | null;
  maps: readonly UnifiedBevMap[]; people: readonly UnifiedBevPerson[];
  unavailable: readonly Readonly<{ entityId: string; stableId: number | null; reason: string; expiresAtMs: number }>[];
  cursor: Readonly<{ run: string; sequence: number; snapshotId: string; producer: JsonRecord; publishedAtUs: number; cohort: Cohort }> | null;
  deadlines: readonly Readonly<{ entityId: string; observedAtUs: number; staleAfterUs: number; expiresAtMs: number }>[];
  sourceCursors: readonly Readonly<{ sourceId: number; run: string; sequence: number; frameId: number | null; epoch: number | null }>[];
  pending: readonly JsonRecord[]; retiredRuns: readonly string[]; eventSequence: number | null; lastReceivedAtMs: number | null;
  diagnostic: Readonly<{ accepted: boolean; reason: string; type: string | null }>;
}>;
export const UNIFIED_BEV_LIMITS: Readonly<{
  cameras: 32; sources: 64; entities: 256; evidencePerEntity: 32; eventsPerCohort: 256; trailPoints: 128; retiredRuns: 4096;
}>;
export class UnifiedBevError extends Error { code: string; constructor(code: string); }
/** Expected digest must identify the authored target selected for this view. */
export function createUnifiedBevBindings(bundle: unknown, authoredSceneSha256: string): UnifiedBevBindings;
/** Returns the registration for raw local geometry/rasters; never person coordinates. */
export function projectUnifiedBevMap(bindings: UnifiedBevBindings, floorplan: unknown): UnifiedBevMap;
export function projectUnifiedBevMapPoint(map: UnifiedBevMap, localXZ: readonly [number, number], heightAboveFloorM?: number): Vec3;
export function createUnifiedBevState(options?: { authoredSceneSha256?: string; maxTrailPoints?: number }): UnifiedBevState;
/** Dispatch calibration-bundle, floorplan_response, tracking, world_snapshot and world_event in wire order. */
export function reduceUnifiedBev(state: UnifiedBevState, message: unknown, receipt: { receivedAtMs: number }): UnifiedBevState;
/** Caller ticks at the next person deadline; no optional rendering work renews expiry. */
export function advanceUnifiedBev(state: UnifiedBevState, nowMs: number): UnifiedBevState;
/** Retains sequence/run fences so reconnect cannot resurrect an older snapshot. */
export function clearUnifiedBevPeople(state: UnifiedBevState, reason?: string): UnifiedBevState;
