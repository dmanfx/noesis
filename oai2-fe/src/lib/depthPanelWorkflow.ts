export type FloorplanGridLayerLike = {
  grid_b64?: unknown;
  grid_shape?: unknown;
};

export type FloorplanResponseLike = {
  type?: unknown;
  request_id?: unknown;
  requestId?: unknown;
  camera_id?: unknown;
  camera?: unknown;
  snapshot_ts?: unknown;
  ts?: unknown;
  served_from_cache?: unknown;
  cache_only?: unknown;
  scene_prior_only?: unknown;
  display_source?: unknown;
  live_floorplan_error?: unknown;
  error?: unknown;
  density?: FloorplanGridLayerLike;
  height?: FloorplanGridLayerLike;
  height_agl?: FloorplanGridLayerLike;
  distance?: FloorplanGridLayerLike;
  gradient?: FloorplanGridLayerLike;
  obstacle_height?: FloorplanGridLayerLike;
  walkable?: FloorplanGridLayerLike;
  scene_static_height_agl?: FloorplanGridLayerLike;
  scene_composite_height_agl?: FloorplanGridLayerLike;
  scene_prior_diagnostic_height_agl?: FloorplanGridLayerLike;
};

export type DepthPanelFloorplanRequest = {
  camera: string;
  requestId: string;
  maxAgeSec: 0;
  gridResM: 0.04;
  maxExtentM: 20;
  cacheOnly: false;
};

export type DepthPanelRefreshAction =
  | {
      kind: 'floorplan';
      request: DepthPanelFloorplanRequest;
    }
  | {
      kind: 'depth-cache';
      cameraId: string;
      strategy: 'cache-only';
      tsMaxUs: number;
    };

export type DepthPanelRefreshResult = {
  handled: boolean;
  action: DepthPanelRefreshAction | null;
  completed?: boolean;
  error?: string;
};

type PendingRefresh = {
  cameraId: string;
  requestId: string;
};

const floorplanLayers = (floorplan: FloorplanResponseLike): Array<FloorplanGridLayerLike | undefined> => [
  floorplan.scene_composite_height_agl,
  floorplan.scene_static_height_agl,
  floorplan.scene_prior_diagnostic_height_agl,
  floorplan.walkable,
  floorplan.obstacle_height,
  floorplan.height_agl,
  floorplan.height,
  floorplan.density,
  floorplan.distance,
  floorplan.gradient,
];

const hasRenderableGridLayer = (layer?: FloorplanGridLayerLike): boolean => {
  if (!layer || typeof layer.grid_b64 !== 'string' || layer.grid_b64.length === 0) return false;
  if (!Array.isArray(layer.grid_shape) || layer.grid_shape.length !== 2) return false;
  return layer.grid_shape.every((dimension) => Number.isSafeInteger(dimension) && Number(dimension) > 0);
};

export const floorplanHasRenderableGrid = (
  floorplan?: FloorplanResponseLike | null,
): boolean => Boolean(
  floorplan
  && !floorplan.error
  && floorplanLayers(floorplan).some(hasRenderableGridLayer)
);

const floorplanSnapshotTsUs = (
  floorplan?: FloorplanResponseLike | null,
): number | null => {
  const candidate = floorplan?.snapshot_ts ?? floorplan?.ts;
  return typeof candidate === 'number' && Number.isSafeInteger(candidate) && candidate > 0
    ? candidate
    : null;
};

export const shouldAdmitFloorplan = (
  existing: FloorplanResponseLike | undefined,
  incoming: FloorplanResponseLike,
): boolean => {
  const existingIsPcf = existing?.scene_prior_only === true
    && existing?.display_source === 'pcf';
  const incomingIsPcf = incoming.scene_prior_only === true
    && incoming.display_source === 'pcf';
  if (incoming.scene_prior_only === true) {
    return incomingIsPcf
      ? floorplanHasRenderableGrid(incoming)
      : !existingIsPcf;
  }
  return false;
};

export const isFloorplanCacheMiss = (
  floorplan?: FloorplanResponseLike | null,
): boolean => String(floorplan?.error ?? '').trim() === 'no_cached_floorplan';

export const shouldRequestCachedFloorplan = ({
  open,
  activeTab,
  selectedCamera,
  floorplan,
}: {
  open: boolean;
  activeTab: string;
  selectedCamera: string;
  floorplan?: FloorplanResponseLike | null;
}): boolean => (
  open
  && Boolean(selectedCamera.trim())
  && (activeTab === 'heatmap' || activeTab === '3d')
  && !floorplanHasRenderableGrid(floorplan)
);

const responseRequestId = (payload: FloorplanResponseLike): string => String(
  payload.request_id ?? payload.requestId ?? '',
).trim();

const responseCameraId = (payload: FloorplanResponseLike): string => String(
  payload.camera_id ?? payload.camera ?? '',
).trim();

const responseError = (payload: FloorplanResponseLike): string => String(
  payload.error ?? '',
).trim();

/**
 * Coordinates a manual static-camera comparison capture without owning its
 * transport. The response is validated and acknowledged, but never becomes a
 * presentation action; the visible panel remains bound to PCF.
 */
export class DepthPanelRefreshCoordinator {
  private sequence = 0;

  private readonly pendingByRequestId = new Map<string, PendingRefresh>();

  private readonly requestIdByCamera = new Map<string, string>();

  constructor(private readonly requestIdFactory?: (cameraId: string, sequence: number) => string) {}

  begin(cameraIdValue: unknown): DepthPanelRefreshAction | null {
    const cameraId = String(cameraIdValue ?? '').trim();
    if (!cameraId || this.requestIdByCamera.has(cameraId)) return null;

    const sequence = ++this.sequence;
    const generated = this.requestIdFactory?.(cameraId, sequence)
      ?? `depth-panel-refresh-${sequence}-${cameraId}`;
    const requestId = String(generated || '').trim();
    if (!requestId || this.pendingByRequestId.has(requestId)) return null;

    const pending = { cameraId, requestId };
    this.pendingByRequestId.set(requestId, pending);
    this.requestIdByCamera.set(cameraId, requestId);
    return {
      kind: 'floorplan',
      request: {
        camera: cameraId,
        requestId,
        maxAgeSec: 0,
        gridResM: 0.04,
        maxExtentM: 20,
        cacheOnly: false,
      },
    };
  }

  handleFloorplanResponse(payloadValue: unknown): DepthPanelRefreshResult {
    if (!payloadValue || typeof payloadValue !== 'object' || Array.isArray(payloadValue)) {
      return { handled: false, action: null };
    }
    const payload = payloadValue as FloorplanResponseLike;
    const requestId = responseRequestId(payload);
    const pending = this.pendingByRequestId.get(requestId);
    if (!requestId || !pending) return { handled: false, action: null };

    this.finish(pending);
    const cameraId = responseCameraId(payload);
    if (cameraId !== pending.cameraId) {
      return { handled: true, action: null, error: 'floorplan_camera_mismatch' };
    }
    const error = responseError(payload);
    if (error) return { handled: true, action: null, error };
    const liveFloorplanError = String(payload.live_floorplan_error ?? '').trim();
    if (liveFloorplanError) {
      return { handled: true, action: null, error: liveFloorplanError };
    }
    if (
      payload.scene_prior_only === true
      || payload.cache_only === true
      || payload.served_from_cache !== false
    ) {
      return { handled: true, action: null, error: 'static_capture_not_fresh' };
    }
    if (!floorplanHasRenderableGrid(payload)) {
      return { handled: true, action: null, error: 'floorplan_not_renderable' };
    }
    const snapshotTs = payload.snapshot_ts;
    if (!Number.isSafeInteger(snapshotTs) || Number(snapshotTs) <= 0) {
      return { handled: true, action: null, error: 'floorplan_snapshot_ts_invalid' };
    }
    return { handled: true, action: null, completed: true };
  }

  cancel(cameraIdValue?: unknown): void {
    if (cameraIdValue === undefined) {
      this.pendingByRequestId.clear();
      this.requestIdByCamera.clear();
      return;
    }
    const cameraId = String(cameraIdValue ?? '').trim();
    const requestId = this.requestIdByCamera.get(cameraId);
    if (!requestId) return;
    this.pendingByRequestId.delete(requestId);
    this.requestIdByCamera.delete(cameraId);
  }

  private finish(pending: PendingRefresh): void {
    this.pendingByRequestId.delete(pending.requestId);
    if (this.requestIdByCamera.get(pending.cameraId) === pending.requestId) {
      this.requestIdByCamera.delete(pending.cameraId);
    }
  }
}
