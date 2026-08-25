export type ResolverCovarianceXZ = [[number, number], [number, number]];

export type ResolverDisplayPoint = { x: number; z: number };

export type ResolverWorldPoint = { x: number; z: number };

export type ResolverPcfSummary = {
  observedConfidence?: number;
  priorId?: string;
  revisionId?: string;
  status?: string;
  insideExtent?: boolean;
  insideAuthoredSpace?: boolean;
  evidenceObserved?: boolean;
  extentOutsideDistanceM?: number;
  boundarySignedDistanceM?: number;
  floorHeightM?: number;
  obstacleClearanceM?: number;
  reasons?: string[];
};

export type ResolverCandidate = {
  id?: string | null;
  kind?: string | null;
  world?: ResolverWorldPoint;
  display?: ResolverDisplayPoint;
  covarianceXZ?: ResolverCovarianceXZ;
  status?: string | null;
  selected?: boolean;
  pcf?: ResolverPcfSummary;
};

export type ResolverPoint = {
  world?: ResolverWorldPoint;
  display?: ResolverDisplayPoint;
  covarianceXZ?: ResolverCovarianceXZ;
};

export type ResolverLegacyPoint = ResolverCandidate & { source?: string | null };

export type ResolverDisagreement = {
  distanceM?: number;
  floorDepthDeltaM?: number;
  innovationM?: number;
  reason?: string;
};

export type ResolverDiagnostics = {
  contract?: string | null;
  version?: number;
  frameId?: number | null;
  sourceId?: number | null;
  sensorId?: number | null;
  worldFrame?: string | null;
  worldFrameRevision?: string | null;
  pcfRevision?: string | null;
  floorplanSnapshotId?: string | null;
  floorplanSnapshotContentSha256?: string | null;
  floorplanCalibrationFingerprint?: string | null;
  floorplanWorldFrame?: string | null;
  floorplanWorldFrameRevision?: string | null;
  floorplanTsUs?: number | null;
  selectedId?: string | null;
  selectedKind?: string | null;
  decision?: string | null;
  reason?: string | null;
  resolved?: ResolverPoint;
  candidates?: ResolverCandidate[];
  legacy?: ResolverLegacyPoint;
  disagreement?: ResolverDisagreement;
};

const MAX_CANDIDATES = 4;
const MAX_TEXT_LENGTH = 96;

const finiteNumber = (value: unknown): value is number => (
  typeof value === 'number' && Number.isFinite(value)
);

const boundedText = (value: unknown): string | null => {
  if (typeof value !== 'string') return null;
  const text = value.trim();
  return text ? text.slice(0, MAX_TEXT_LENGTH) : null;
};

const validPoint = (value: unknown): value is ResolverDisplayPoint => (
  Boolean(value)
  && typeof value === 'object'
  && finiteNumber((value as ResolverDisplayPoint).x)
  && finiteNumber((value as ResolverDisplayPoint).z)
);

const validWorldPoint = (value: unknown): value is ResolverWorldPoint => validPoint(value);

export const validResolverCovariance = (value: unknown): value is ResolverCovarianceXZ => {
  if (!Array.isArray(value) || value.length !== 2) return false;
  if (!Array.isArray(value[0]) || value[0].length !== 2) return false;
  if (!Array.isArray(value[1]) || value[1].length !== 2) return false;
  const [a, b, c, d] = [value[0][0], value[0][1], value[1][0], value[1][1]];
  if (![a, b, c, d].every(finiteNumber)) return false;
  if (Math.abs(Number(b) - Number(c)) > 1e-4) return false;
  const determinant = Number(a) * Number(d) - Number(b) * Number(c);
  return Number(a) >= -1e-5 && Number(d) >= -1e-5 && determinant >= -1e-5;
};

const validResolverPoint = (value: unknown): value is ResolverPoint => {
  if (!value || typeof value !== 'object') return false;
  const point = value as ResolverPoint;
  if (!validWorldPoint(point.world) || !validPoint(point.display)) return false;
  if (point.covarianceXZ !== undefined && !validResolverCovariance(point.covarianceXZ)) return false;
  return true;
};

const validResolverPcf = (value: unknown): value is ResolverPcfSummary => {
  if (!value || typeof value !== 'object') return false;
  const pcf = value as ResolverPcfSummary;
  for (const key of [
    'observedConfidence',
    'extentOutsideDistanceM',
    'boundarySignedDistanceM',
    'floorHeightM',
    'obstacleClearanceM',
  ] as const) {
    if (pcf[key] !== undefined && !finiteNumber(pcf[key])) return false;
  }
  for (const key of ['insideExtent', 'insideAuthoredSpace', 'evidenceObserved'] as const) {
    if (pcf[key] !== undefined && typeof pcf[key] !== 'boolean') return false;
  }
  for (const key of ['priorId', 'revisionId', 'status'] as const) {
    if (pcf[key] !== undefined && boundedText(pcf[key]) === null) return false;
  }
  if (pcf.reasons !== undefined && (
    !Array.isArray(pcf.reasons)
    || pcf.reasons.length > 8
    || !pcf.reasons.every((reason) => boundedText(reason) !== null)
  )) return false;
  return true;
};

const validResolverCandidate = (value: unknown): value is ResolverCandidate => {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as ResolverCandidate;
  if (!validResolverPoint(candidate)) return false;
  if (candidate.id !== undefined && candidate.id !== null && boundedText(candidate.id) === null) return false;
  if (candidate.kind !== undefined && candidate.kind !== null && boundedText(candidate.kind) === null) return false;
  if (candidate.status !== undefined && candidate.status !== null && boundedText(candidate.status) === null) return false;
  if (candidate.selected !== undefined && typeof candidate.selected !== 'boolean') return false;
  if (candidate.pcf !== undefined && !validResolverPcf(candidate.pcf)) return false;
  return true;
};

const validResolverDisagreement = (value: unknown): value is ResolverDisagreement => {
  if (!value || typeof value !== 'object') return false;
  const disagreement = value as ResolverDisagreement;
  for (const key of ['distanceM', 'floorDepthDeltaM', 'innovationM'] as const) {
    if (disagreement[key] !== undefined && !finiteNumber(disagreement[key])) return false;
  }
  if (disagreement.reason !== undefined && boundedText(disagreement.reason) === null) return false;
  return Object.keys(disagreement).length > 0;
};

const sameOptionalIdentity = (diagnostic: unknown, payload: unknown): boolean => {
  if (payload === undefined || payload === null || payload === '') return true;
  if (diagnostic === undefined || diagnostic === null || diagnostic === '') return false;
  return String(diagnostic) === String(payload);
};

/**
 * Admit diagnostics only when they belong to the currently rendered BEV
 * point and its exact producer/world/floorplan revision. This function never
 * consults a previous payload and intentionally returns null on any missing
 * or ambiguous identity.
 */
export const admitResolverDiagnostics = (
  point: unknown,
  payload: unknown,
): ResolverDiagnostics | null => {
  if (!point || typeof point !== 'object' || !payload || typeof payload !== 'object') return null;
  const raw = (point as { resolverDiagnostics?: unknown }).resolverDiagnostics;
  if (!raw || typeof raw !== 'object') return null;
  const diagnostic = raw as ResolverDiagnostics;
  if (diagnostic.contract !== 'noesis.world_resolver_diagnostics') return null;
  if (diagnostic.version !== 1) return null;

  const frameId = (payload as { frameId?: unknown }).frameId;
  if (!Number.isInteger(diagnostic.frameId) || diagnostic.frameId < 0 || !Number.isInteger(frameId) || diagnostic.frameId !== frameId) return null;
  const pointFrameId = (point as { frameId?: unknown }).frameId;
  if (pointFrameId !== undefined && pointFrameId !== null && pointFrameId !== frameId) return null;
  const rawWorldFrame = (payload as { canonicalWorldFrame?: unknown }).canonicalWorldFrame;
  const rawWorldRevision = (payload as { canonicalWorldFrameRevision?: unknown }).canonicalWorldFrameRevision;
  const canonicalWorldFrame = typeof rawWorldFrame === 'string' ? rawWorldFrame.trim() : '';
  const canonicalWorldRevision = typeof rawWorldRevision === 'string' ? rawWorldRevision.trim() : '';
  if (!canonicalWorldFrame || !canonicalWorldRevision) return null;
  if (!boundedText(diagnostic.worldFrame) || !boundedText(diagnostic.worldFrameRevision)) return null;
  if (diagnostic.worldFrame !== canonicalWorldFrame) return null;
  if (diagnostic.worldFrameRevision !== canonicalWorldRevision) return null;

  const payloadSourceId = (payload as { sourceId?: unknown }).sourceId;
  if (!Number.isInteger(diagnostic.sensorId) || !Number.isInteger(payloadSourceId)) return null;
  if (diagnostic.sensorId !== payloadSourceId) return null;
  if (!Number.isInteger(diagnostic.sourceId) || Number(diagnostic.sourceId) < 0) return null;
  if (diagnostic.floorplanWorldFrame !== undefined && diagnostic.floorplanWorldFrame !== null) {
    if (diagnostic.floorplanWorldFrame !== canonicalWorldFrame) return null;
  }
  if (
    diagnostic.floorplanWorldFrameRevision !== undefined
    && diagnostic.floorplanWorldFrameRevision !== null
    && diagnostic.floorplanWorldFrameRevision !== canonicalWorldRevision
  ) return null;

  const identityPairs: Array<[unknown, unknown]> = [
    [diagnostic.floorplanSnapshotId, (payload as { floorplanSnapshotId?: unknown }).floorplanSnapshotId],
    [diagnostic.floorplanSnapshotContentSha256, (payload as { floorplanSnapshotContentSha256?: unknown }).floorplanSnapshotContentSha256],
    [diagnostic.floorplanCalibrationFingerprint, (payload as { floorplanCalibrationFingerprint?: unknown }).floorplanCalibrationFingerprint],
  ];
  if (identityPairs.some(([diag, current]) => !sameOptionalIdentity(diag, current))) return null;

  if (diagnostic.resolved !== undefined && !validResolverPoint(diagnostic.resolved)) return null;
  if (diagnostic.legacy !== undefined && !validResolverCandidate(diagnostic.legacy)) return null;
  if (diagnostic.candidates !== undefined) {
    if (!Array.isArray(diagnostic.candidates) || diagnostic.candidates.length > MAX_CANDIDATES) return null;
    if (!diagnostic.candidates.every(validResolverCandidate)) return null;
  }
  if (diagnostic.disagreement !== undefined && !validResolverDisagreement(diagnostic.disagreement)) return null;
  const candidates = Array.isArray(diagnostic.candidates) ? diagnostic.candidates : [];
  if (!diagnostic.resolved && candidates.length === 0 && !diagnostic.legacy) return null;
  return {
    ...diagnostic,
    contract: 'noesis.world_resolver_diagnostics',
    selectedId: boundedText(diagnostic.selectedId),
    selectedKind: boundedText(diagnostic.selectedKind),
    decision: boundedText(diagnostic.decision),
    reason: boundedText(diagnostic.reason),
    candidates: candidates.slice(0, MAX_CANDIDATES),
  };
};

export const resolverDiagnosticDisplayPoint = (
  point: ResolverPoint | null | undefined,
): ResolverDisplayPoint | null => {
  if (!point) return null;
  if (validPoint(point.display)) return point.display;
  return null;
};

/**
 * Mirror a successfully transmitted comparison request immediately. The
 * backend toggle_update remains authoritative and can still reconcile the
 * state, but the controlled checkbox must not appear to ignore a valid click
 * while that acknowledgement is in flight.
 */
export const requestLocalizationDetailsChange = (
  enabled: boolean,
  send: (next: boolean) => boolean,
  update: (next: boolean) => void,
): boolean => {
  if (!send(enabled)) return false;
  update(enabled);
  return true;
};

export const resolverDiagnosticsMaxCandidates = MAX_CANDIDATES;
