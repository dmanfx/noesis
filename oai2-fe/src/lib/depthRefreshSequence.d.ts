export type ExactFloorplanContinuation = {
  identityKey: string;
  request: {
    camera: string;
    requestId: string;
    maxAgeSec: number;
    gridResM: number;
    maxExtentM: number;
    cacheOnly: false;
    snapshotRef: string;
    snapshotId: string;
    snapshotContentSha256: string;
  };
};

export function exactFloorplanContinuation(
  message: unknown,
  options?: {
    gridResM?: number;
    maxExtentM?: number;
    nowMs?: number;
  },
): ExactFloorplanContinuation | null;

export type PendingExactFloorplan = {
  requestId: string;
  snapshotRef: string;
  snapshotId: string;
  snapshotContentSha256: string;
};

export type ExactFloorplanResponseOutcome =
  | { kind: 'unrelated' }
  | { kind: 'commit' }
  | { kind: 'error'; error: string };

export function exactFloorplanResponseOutcome(
  pending: PendingExactFloorplan | null | undefined,
  message: unknown,
  options?: { renderable?: boolean },
): ExactFloorplanResponseOutcome;

export function floorplanMatchesActiveDepth(
  activeSnapshotId: unknown,
  floorplan: unknown,
): boolean;

export type CachedSnapshotPairOutcome =
  | { kind: 'waiting'; waitingFor: 'depth' | 'floorplan' }
  | { kind: 'commit' }
  | { kind: 'error'; error: string };

export function cachedSnapshotPairOutcome(
  depth: unknown,
  floorplan: unknown,
  options?: { renderable?: boolean },
): CachedSnapshotPairOutcome;
