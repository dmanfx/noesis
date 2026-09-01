export type CurrentBevHeadState = {
  x: number;
  y: number;
  lastSeen: number;
  stableId?: string;
  colorId: number;
  worldAdmission?: string;
};

export function historyKeyForBevIdentity(identity?: {
  historyKey?: string | null;
  stableId?: number | null;
  trackerId?: number | null;
  trackerLifecycleGeneration?: number | null;
}): string | null;

export function sourceTimelineKeyForBevPayload(payload?: {
  sourceId?: number | null;
  sourceEpoch?: number | null;
  cohort?: {
    source_id?: number | null;
    source_epoch?: number | null;
  } | null;
} | null): string | null;

export function resetBevVisualContinuityState<Head, Trail, Clock extends object>(args: {
  heads: Map<string, Head>;
  trails: Map<string, Trail>;
  frameCounterRef: { current: number };
  clockRef: { current: Clock };
}): void;

export function purgeCurrentBevHeadState(
  state: Map<string, CurrentBevHeadState>,
  historyKey?: string | null,
): boolean;

export function purgeCurrentBevHeadsForDropped(
  state: Map<string, CurrentBevHeadState>,
  droppedFootpoints?: Array<{
    historyKey?: string | null;
    stableId?: number | null;
    trackerId?: number | null;
    trackerLifecycleGeneration?: number | null;
  }> | null,
): number;

export function purgeCurrentBevHeadsMissingFromCohort(
  state: Map<string, CurrentBevHeadState>,
  seenHistoryKeys: Set<string>,
): number;

export function updateCurrentBevHeadState(args: {
  state: Map<string, CurrentBevHeadState>;
  point?: { worldAdmission?: string | null } | null;
  resolved?: { x: number; y: number } | null;
  nowMs: number;
  historyKey?: string | null;
  displayId?: number | null;
  colorId: number;
}): boolean;
