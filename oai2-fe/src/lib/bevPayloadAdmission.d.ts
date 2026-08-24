export type BevAdmissionResult<T> = {
  admitted: boolean;
  reason: string;
  payload: T | null;
};

export function bevCameraId(payload: unknown): string;
export function bevCohort(payload: unknown): {
  sourceId: number;
  frameId: number;
  observedAtUs: number;
  sequence: number;
  outboundSubmissionId: number;
} | null;
export function admitBevFrame<T extends Record<string, unknown>>(
  previous: T | undefined,
  incoming: T,
): BevAdmissionResult<T>;
export function admitBevStatus<T extends Record<string, unknown>>(
  previous: T | undefined,
  incoming: T,
): BevAdmissionResult<T>;
export function clearBevForStatus<T extends Record<string, unknown>>(
  previous: T | undefined,
  status: T,
): T;
export function bevMatchesFloorplan(bev: unknown, floorplan: unknown): boolean;
