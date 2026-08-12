import {
  DEPTH_BULK_MAX_ACTIVE_REQUESTS,
  DEPTH_BULK_REQUEST_TIMEOUT_MS,
} from './depthBulkLimits';
import type { DepthDiagnosticsSummary } from './depthDiagnostics';
import { normalizeCameraIntrinsics } from './depthNormals';

export type DepthBulkComponentName = 'depth' | 'conf' | 'mask' | 'rgb';

export type DepthBulkComponentDescriptor = {
  component: DepthBulkComponentName;
  dtype: '<f4' | '|u1';
  shape: number[];
  byte_count: number;
  sha256: string;
  url: string;
};

export type DepthBulkSnapshotDescriptor = {
  contract: 'noesis.depth.bulk_snapshot';
  contract_version: 1;
  ts: number;
  shape: [number, number];
  snapshot_id: string;
  snapshot_ref: string;
  content_sha256: string;
  role: string;
  fusion_level: string;
  components: {
    depth: DepthBulkComponentDescriptor;
    conf: DepthBulkComponentDescriptor;
    mask: DepthBulkComponentDescriptor;
    rgb?: DepthBulkComponentDescriptor;
  };
  normals: {
    mode: 'client_derived_depth_gradient_v1';
    space: 'camera';
    dtype: 'float32';
  };
  capture_event?: Record<string, unknown>;
  capture_event_evidence_sha256?: string;
};

export type LoadedDepthBulkSnapshot = {
  ts: number;
  shape: [number, number];
  depth: Float32Array;
  conf: Float32Array;
  mask: Uint8Array;
  rgb?: Uint8Array;
  rgbShape?: [number, number, number];
  normals: Float32Array;
  normalsShape: [number, number, number];
  surfaceNormals?: Int8Array;
  surfaceNormalsShape?: [number, number, number];
  snapshotId: string;
  snapshotRef: string;
  snapshotContentSha256: string;
  diagnostics: DepthDiagnosticsSummary;
  transferBytes: number;
  transferDurationMs: number;
};

type WorkerSuccess = {
  type: 'loaded';
  requestId: string;
  snapshot: LoadedDepthBulkSnapshot;
};

type WorkerFailure = {
  type: 'error';
  requestId: string;
  error: string;
};

type PendingRequest = {
  cameraId: string;
  resolve: (snapshot: LoadedDepthBulkSnapshot) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
};

export type DepthBulkLoadOptions = {
  includeRgb?: boolean;
  /** Absolute performance.now() deadline inherited from the WS request. */
  deadlineAtMs?: number;
  /** Camera calibration as [fx, fy, cx, cy] or a row-major 3x3 K matrix. */
  intrinsics?: number[] | null;
};

let worker: Worker | null = null;
let nextRequestId = 0;
const pending = new Map<string, PendingRequest>();

const ensureWorker = (): Worker => {
  if (worker) return worker;
  const created = new Worker(
    new URL('../workers/depthBulkWorker.ts', import.meta.url),
    { type: 'module', name: 'noesis-depth-bulk' },
  );
  created.onmessage = (event: MessageEvent<WorkerSuccess | WorkerFailure>) => {
    const message = event.data;
    const request = pending.get(message?.requestId);
    if (!request) return;
    pending.delete(message.requestId);
    clearTimeout(request.timeout);
    if (message.type === 'loaded') {
      request.resolve(message.snapshot);
    } else {
      request.reject(new Error(message.error || 'depth_bulk_worker_failed'));
    }
  };
  created.onerror = (event: ErrorEvent) => {
    const error = new Error(event.message || 'depth_bulk_worker_crashed');
    for (const request of pending.values()) {
      clearTimeout(request.timeout);
      request.reject(error);
    }
    pending.clear();
    created.terminate();
    worker = null;
  };
  worker = created;
  return created;
};

export const loadDepthBulkSnapshot = (
  descriptor: unknown,
  expectedCameraId: string,
  options: DepthBulkLoadOptions = {},
): Promise<LoadedDepthBulkSnapshot> => {
  const requestId = `depth-bulk-${Date.now()}-${++nextRequestId}`;
  return new Promise((resolve, reject) => {
    const includeRgb = options.includeRgb ?? false;
    if (typeof includeRgb !== 'boolean') {
      reject(new Error('depth_bulk_include_rgb_invalid'));
      return;
    }
    const intrinsics = normalizeCameraIntrinsics(options.intrinsics);
    if (!intrinsics) {
      reject(new Error('depth_bulk_intrinsics_missing'));
      return;
    }
    let timeoutMs = DEPTH_BULK_REQUEST_TIMEOUT_MS;
    if (options.deadlineAtMs !== undefined) {
      const deadlineAtMs = Number(options.deadlineAtMs);
      if (!Number.isFinite(deadlineAtMs)) {
        reject(new Error('depth_bulk_deadline_invalid'));
        return;
      }
      const remainingMs = Math.floor(deadlineAtMs - performance.now());
      if (remainingMs < 1) {
        reject(new Error('depth_bulk_request_deadline_exceeded'));
        return;
      }
      timeoutMs = Math.min(DEPTH_BULK_REQUEST_TIMEOUT_MS, remainingMs);
    }
    let activeWorker: Worker;
    try {
      activeWorker = ensureWorker();
    } catch (error) {
      reject(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    for (const [priorRequestId, request] of Array.from(pending.entries())) {
      if (request.cameraId !== expectedCameraId) continue;
      pending.delete(priorRequestId);
      clearTimeout(request.timeout);
      try {
        activeWorker.postMessage({ type: 'cancel', requestId: priorRequestId });
      } catch {
        // The worker may already have terminated; the prior request still fails closed.
      }
      request.reject(new Error('depth_bulk_request_superseded'));
    }
    if (pending.size >= DEPTH_BULK_MAX_ACTIVE_REQUESTS) {
      reject(new Error('depth_bulk_request_capacity_exceeded'));
      return;
    }
    const timeout = setTimeout(() => {
      pending.delete(requestId);
      try {
        activeWorker.postMessage({ type: 'cancel', requestId });
      } catch {
        // The worker may already have terminated. The request still fails closed.
      }
      reject(new Error('depth_bulk_transfer_timeout'));
    }, timeoutMs);
    pending.set(requestId, {
      cameraId: expectedCameraId,
      resolve,
      reject,
      timeout,
    });
    try {
      activeWorker.postMessage({
        type: 'load',
        requestId,
        descriptor,
        expectedCameraId,
        baseUrl: window.location.origin,
        includeRgb,
        intrinsics,
        timeoutMs,
      });
    } catch (error) {
      pending.delete(requestId);
      clearTimeout(timeout);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
};
