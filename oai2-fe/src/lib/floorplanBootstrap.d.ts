export type FloorplanBootstrapRequest = {
  camera: string;
  requestId: string;
  maxAgeSec: number;
  gridResM: number;
  maxExtentM: number;
  cacheOnly: boolean;
  scenePriorOnly: boolean;
};

export type FloorplanBootstrapAction = {
  request: FloorplanBootstrapRequest;
  delayMs: number;
};

export type FloorplanBootstrapResult = {
  handled: boolean;
  action: FloorplanBootstrapAction | null;
  completedCamera?: string | null;
  failedCamera?: string | null;
  error?: string | null;
};

export type FloorplanBootstrapSnapshot = {
  run: number;
  queue: string[];
  active: null | {
    camera: string;
    phase: 'cache' | 'fresh';
    retryCount: number;
    requestId: string;
  };
  pendingRestart: string[] | null;
};

export class FloorplanBootstrapCoordinator {
  constructor(options?: {
    requestIdPrefix?: string;
    maxTransientRetries?: number;
    retryDelayMs?: number;
  });
  restart(cameras: string[]): FloorplanBootstrapAction | null;
  cancel(): void;
  isCurrentRequest(requestId: string): boolean;
  handleResponse(payload: unknown, options?: { renderable?: boolean }): FloorplanBootstrapResult;
  snapshot(): FloorplanBootstrapSnapshot;
}
