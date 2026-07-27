/// <reference lib="webworker" />

import type {
  DepthBulkComponentDescriptor,
  DepthBulkComponentName,
  DepthBulkSnapshotDescriptor,
  LoadedDepthBulkSnapshot,
} from '../lib/depthBulkClient';
import {
  DEPTH_BULK_MAX_ACTIVE_ALLOCATION_BYTES,
  DEPTH_BULK_MAX_ACTIVE_REQUESTS,
  DEPTH_BULK_MAX_COMPONENT_BYTES,
  DEPTH_BULK_MAX_PIXELS,
  DEPTH_BULK_MAX_SNAPSHOT_BYTES,
  DEPTH_BULK_REQUEST_TIMEOUT_MS,
} from '../lib/depthBulkLimits';
import { deriveDepthDiagnostics } from '../lib/depthDiagnostics';
import {
  deriveCalibratedDepthNormals,
  normalizeCameraIntrinsics,
} from '../lib/depthNormals';
import { sha256Hex } from '../lib/sha256';

declare const self: DedicatedWorkerGlobalScope;

const SHA256_RE = /^[0-9a-f]{64}$/;
const REQUIRED_COMPONENTS: readonly DepthBulkComponentName[] = ['depth', 'conf', 'mask'];
const ALLOWED_COMPONENTS = new Set<DepthBulkComponentName>([
  ...REQUIRED_COMPONENTS,
  'rgb',
]);
const COMPONENT_DESCRIPTOR_KEYS = new Set([
  'component',
  'dtype',
  'shape',
  'byte_count',
  'sha256',
  'url',
]);
const SNAPSHOT_DESCRIPTOR_KEYS = new Set([
  'contract',
  'contract_version',
  'ts',
  'shape',
  'snapshot_id',
  'snapshot_ref',
  'content_sha256',
  'role',
  'fusion_level',
  'components',
  'normals',
  'capture_event',
  'capture_event_evidence_sha256',
]);
type ActiveRequest = {
  controller: AbortController;
  cameraId: string;
  reservedBytes: number;
  deadlineMs: number;
  timeout: ReturnType<typeof setTimeout>;
  timedOut: boolean;
  cancelled: boolean;
};

const controllers = new Map<string, ActiveRequest>();
let activeAllocationBytes = 0;

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === 'object' && value !== null && !Array.isArray(value)
);

const positiveSafeInteger = (value: unknown): value is number => (
  Number.isSafeInteger(value) && Number(value) > 0
);

const hasExactKeys = (value: Record<string, unknown>, allowed: Set<string>): boolean => (
  Object.keys(value).every((key) => allowed.has(key))
);

const assertNoInlineBinary = (value: unknown, path = '$'): void => {
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoInlineBinary(item, `${path}[${index}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, item] of Object.entries(value)) {
    if (key.toLowerCase().endsWith('_b64')) {
      throw new Error(`depth_bulk_inline_binary_forbidden:${path}.${key}`);
    }
    assertNoInlineBinary(item, `${path}.${key}`);
  }
};

const validateShape = (value: unknown, rank: number): number[] => {
  if (!Array.isArray(value) || value.length !== rank || !value.every(positiveSafeInteger)) {
    throw new Error('depth_bulk_component_shape_invalid');
  }
  return value.map(Number);
};

const expectedByteCount = (shape: number[], dtype: '<f4' | '|u1'): number => {
  const elements = shape.reduce((left, right) => left * right, 1);
  const bytes = elements * (dtype === '<f4' ? 4 : 1);
  if (
    !Number.isSafeInteger(bytes)
    || bytes <= 0
    || bytes > DEPTH_BULK_MAX_COMPONENT_BYTES
  ) {
    throw new Error('depth_bulk_component_size_invalid');
  }
  return bytes;
};

const validateComponent = (
  raw: unknown,
  expectedName: DepthBulkComponentName,
  expectedShape: number[],
): DepthBulkComponentDescriptor => {
  if (
    !isRecord(raw)
    || !hasExactKeys(raw, COMPONENT_DESCRIPTOR_KEYS)
    || raw.component !== expectedName
  ) {
    throw new Error(`depth_bulk_${expectedName}_descriptor_invalid`);
  }
  const dtype = raw.dtype;
  const expectedDtype: '<f4' | '|u1' = expectedName === 'depth' || expectedName === 'conf'
    ? '<f4'
    : '|u1';
  if (dtype !== expectedDtype) {
    throw new Error(`depth_bulk_${expectedName}_dtype_invalid`);
  }
  const rank = expectedName === 'rgb' ? 3 : 2;
  const shape = validateShape(raw.shape, rank);
  if (
    shape[0] !== expectedShape[0]
    || shape[1] !== expectedShape[1]
    || (expectedName === 'rgb' && shape[2] !== 3)
  ) {
    throw new Error(`depth_bulk_${expectedName}_shape_mismatch`);
  }
  const bytes = expectedByteCount(shape, expectedDtype);
  if (raw.byte_count !== bytes || !positiveSafeInteger(raw.byte_count)) {
    throw new Error(`depth_bulk_${expectedName}_byte_count_mismatch`);
  }
  if (typeof raw.sha256 !== 'string' || !SHA256_RE.test(raw.sha256)) {
    throw new Error(`depth_bulk_${expectedName}_digest_invalid`);
  }
  if (
    typeof raw.url !== 'string'
    || raw.url.length > 2048
    || !raw.url.startsWith('/api/v1/depth/snapshots/')
    || raw.url.startsWith('//')
    || raw.url.includes('\\')
    || /[\u0000-\u001f\u007f]/.test(raw.url)
  ) {
    throw new Error(`depth_bulk_${expectedName}_url_invalid`);
  }
  return raw as DepthBulkComponentDescriptor;
};

const validateDescriptor = (raw: unknown): DepthBulkSnapshotDescriptor => {
  assertNoInlineBinary(raw);
  if (
    !isRecord(raw)
    || !hasExactKeys(raw, SNAPSHOT_DESCRIPTOR_KEYS)
    || raw.contract !== 'noesis.depth.bulk_snapshot'
    || raw.contract_version !== 1
  ) {
    throw new Error('depth_bulk_contract_invalid');
  }
  const shape = validateShape(raw.shape, 2) as [number, number];
  const pixelCount = shape[0] * shape[1];
  if (
    !Number.isSafeInteger(pixelCount)
    || pixelCount > DEPTH_BULK_MAX_PIXELS
    || !positiveSafeInteger(raw.ts)
    || typeof raw.snapshot_id !== 'string'
    || !raw.snapshot_id
    || raw.snapshot_id.length > 160
    || typeof raw.snapshot_ref !== 'string'
    || !raw.snapshot_ref
    || raw.snapshot_ref.length > 512
    || raw.snapshot_ref.startsWith('/')
    || raw.snapshot_ref.includes('\\')
    || raw.snapshot_ref.split('/').some((part) => !part || part === '.' || part === '..')
    || typeof raw.content_sha256 !== 'string'
    || !SHA256_RE.test(raw.content_sha256)
    || raw.role !== 'capture_event_fused'
    || raw.fusion_level !== 'intra_capture'
    || !isRecord(raw.components)
  ) {
    throw new Error('depth_bulk_snapshot_identity_invalid');
  }
  const components = raw.components;
  const componentNames = Object.keys(components);
  if (
    !REQUIRED_COMPONENTS.every((component) => componentNames.includes(component))
    || componentNames.some((component) => !ALLOWED_COMPONENTS.has(component as DepthBulkComponentName))
  ) {
    throw new Error('depth_bulk_component_set_invalid');
  }
  let transferBytes = 0;
  for (const component of componentNames as DepthBulkComponentName[]) {
    const validated = validateComponent(components[component], component, shape);
    transferBytes += validated.byte_count;
  }
  if (
    !Number.isSafeInteger(transferBytes)
    || transferBytes > DEPTH_BULK_MAX_SNAPSHOT_BYTES
  ) {
    throw new Error('depth_bulk_snapshot_size_invalid');
  }
  if (
    !isRecord(raw.normals)
    || Object.keys(raw.normals).length !== 3
    || raw.normals.mode !== 'client_derived_depth_gradient_v1'
    || raw.normals.space !== 'camera'
    || raw.normals.dtype !== 'float32'
  ) {
    throw new Error('depth_bulk_normals_policy_invalid');
  }
  const hasCaptureEvent = raw.capture_event !== undefined;
  const hasCaptureDigest = raw.capture_event_evidence_sha256 !== undefined;
  if (
    hasCaptureEvent !== hasCaptureDigest
    || (hasCaptureEvent && !isRecord(raw.capture_event))
    || (
      hasCaptureDigest
      && (
        typeof raw.capture_event_evidence_sha256 !== 'string'
        || !SHA256_RE.test(raw.capture_event_evidence_sha256)
      )
    )
  ) {
    throw new Error('depth_bulk_capture_evidence_invalid');
  }
  return raw as DepthBulkSnapshotDescriptor;
};

const CAPTURE_EVENT_FLOAT_PATHS = new Set([
  '$.parameters.burst_seconds',
  '$.parameters.depth_agreement_m',
]);

const pythonAsciiJsonString = (value: string): string => JSON.stringify(value)
  .replace(/[\u0080-\uffff]/g, (character) => (
    `\\u${character.charCodeAt(0).toString(16).padStart(4, '0')}`
  ));

const pythonFloatRepresentation = (value: number): string => {
  if (!Number.isFinite(value)) throw new Error('depth_bulk_capture_evidence_invalid');
  if (Object.is(value, -0)) return '-0.0';
  if (Number.isInteger(value)) return `${value}.0`;
  const absolute = Math.abs(value);
  if (absolute !== 0 && absolute < 1e-4) {
    return value.toExponential().replace(
      /e([+-]?)([0-9]+)$/,
      (_match, sign: string, digits: string) => (
        `e${sign === '-' ? '-' : '+'}${digits.padStart(2, '0')}`
      ),
    );
  }
  return value.toString();
};

const canonicalPythonJson = (value: unknown, path = '$'): string => {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'string') return pythonAsciiJsonString(value);
  if (typeof value === 'number') {
    if (CAPTURE_EVENT_FLOAT_PATHS.has(path)) {
      return pythonFloatRepresentation(value);
    }
    if (!Number.isSafeInteger(value)) {
      throw new Error('depth_bulk_capture_evidence_invalid');
    }
    return String(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item, index) => canonicalPythonJson(item, `${path}[${index}]`)).join(',')}]`;
  }
  if (!isRecord(value)) throw new Error('depth_bulk_capture_evidence_invalid');
  const keys = Object.keys(value).sort();
  if (keys.some((key) => !key)) throw new Error('depth_bulk_capture_evidence_invalid');
  return `{${keys.map((key) => (
    `${pythonAsciiJsonString(key)}:${canonicalPythonJson(value[key], `${path}.${key}`)}`
  )).join(',')}}`;
};

const validateCaptureEvidenceDigest = async (
  descriptor: DepthBulkSnapshotDescriptor,
): Promise<void> => {
  if (!descriptor.capture_event || !descriptor.capture_event_evidence_sha256) return;
  const canonical = new TextEncoder().encode(
    canonicalPythonJson(descriptor.capture_event),
  );
  const recomputed = sha256Hex(canonical);
  if (recomputed !== descriptor.capture_event_evidence_sha256) {
    throw new Error('depth_bulk_capture_evidence_digest_mismatch');
  }
};

const decodePathSegment = (value: string): string => {
  try {
    const decoded = decodeURIComponent(value);
    if (!decoded || decoded === '.' || decoded === '..' || decoded.includes('/') || decoded.includes('\\')) {
      throw new Error('invalid segment');
    }
    return decoded;
  } catch {
    throw new Error('depth_bulk_component_url_invalid');
  }
};

const resolveComponentUrl = (
  descriptor: DepthBulkComponentDescriptor,
  expectedCameraId: string,
  snapshotId: string,
  snapshotRef: string,
  snapshotContentSha256: string,
  baseUrl: string,
): URL => {
  const base = new URL(baseUrl);
  if (base.origin !== self.location.origin) {
    throw new Error('depth_bulk_cross_origin_base_forbidden');
  }
  const url = new URL(descriptor.url, base);
  if (url.origin !== base.origin || url.username || url.password || url.hash) {
    throw new Error('depth_bulk_cross_origin_component_forbidden');
  }
  const segments = url.pathname.split('/');
  if (
    segments.length !== 9
    || segments[0] !== ''
    || segments[1] !== 'api'
    || segments[2] !== 'v1'
    || segments[3] !== 'depth'
    || segments[4] !== 'snapshots'
    || segments[7] !== 'components'
  ) {
    throw new Error('depth_bulk_component_url_invalid');
  }
  if (decodePathSegment(segments[5]) !== expectedCameraId) {
    throw new Error('depth_bulk_component_url_camera_mismatch');
  }
  if (decodePathSegment(segments[6]) !== snapshotId) {
    throw new Error('depth_bulk_component_url_snapshot_mismatch');
  }
  if (decodePathSegment(segments[8]) !== descriptor.component) {
    throw new Error('depth_bulk_component_url_role_mismatch');
  }
  const queryKeys: string[] = [];
  url.searchParams.forEach((_value, key) => queryKeys.push(key));
  if (
    queryKeys.length !== 2
    || url.searchParams.getAll('snapshot_ref').length !== 1
    || url.searchParams.get('snapshot_ref') !== snapshotRef
    || url.searchParams.getAll('content_sha256').length !== 1
    || url.searchParams.get('content_sha256') !== snapshotContentSha256
  ) {
    throw new Error('depth_bulk_component_url_identity_mismatch');
  }
  return url;
};

const cancelResponseBody = async (response: Response, error: string): Promise<never> => {
  try {
    await response.body?.cancel(error);
  } catch {
    // Preserve the validation error; cancellation is only for prompt lease release.
  }
  throw new Error(error);
};

const assertBeforeDeadline = (deadlineMs: number): void => {
  if (!Number.isFinite(deadlineMs) || performance.now() >= deadlineMs) {
    throw new Error('depth_bulk_transfer_timeout');
  }
};

const fetchComponent = async (
  descriptor: DepthBulkComponentDescriptor,
  expectedCameraId: string,
  snapshotId: string,
  snapshotRef: string,
  snapshotContentSha256: string,
  baseUrl: string,
  signal: AbortSignal,
  deadlineMs: number,
): Promise<Uint8Array> => {
  assertBeforeDeadline(deadlineMs);
  const url = resolveComponentUrl(
    descriptor,
    expectedCameraId,
    snapshotId,
    snapshotRef,
    snapshotContentSha256,
    baseUrl,
  );
  const response = await fetch(url, {
    method: 'GET',
    credentials: 'same-origin',
    cache: 'no-store',
    redirect: 'error',
    signal,
    headers: { Accept: 'application/octet-stream' },
  });
  assertBeforeDeadline(deadlineMs);
  if (response.status !== 200 || response.redirected || !response.body) {
    return cancelResponseBody(response, `depth_bulk_component_http_${response.status}`);
  }
  const contentType = (response.headers.get('content-type') || '')
    .split(';', 1)[0]
    .trim()
    .toLowerCase();
  const cacheDirectives = (response.headers.get('cache-control') || '')
    .split(',')
    .map((directive) => directive.trim().split('=', 1)[0].toLowerCase());
  const contentLengthHeader = response.headers.get('content-length') || '';
  const contentLength = /^[0-9]+$/.test(contentLengthHeader)
    ? Number(contentLengthHeader)
    : Number.NaN;
  const contentRange = (response.headers.get('content-range') || '').trim();
  const contentEncoding = (response.headers.get('content-encoding') || '').trim().toLowerCase();
  if (contentType !== 'application/octet-stream') {
    return cancelResponseBody(response, 'depth_bulk_component_content_type_invalid');
  }
  if (!cacheDirectives.includes('no-store')) {
    return cancelResponseBody(response, 'depth_bulk_component_cache_policy_invalid');
  }
  if (!Number.isSafeInteger(contentLength) || contentLength !== descriptor.byte_count) {
    return cancelResponseBody(response, 'depth_bulk_component_content_length_mismatch');
  }
  if (contentRange) {
    return cancelResponseBody(response, 'depth_bulk_component_content_range_invalid');
  }
  if (contentEncoding && contentEncoding !== 'identity') {
    return cancelResponseBody(response, 'depth_bulk_component_content_encoding_invalid');
  }
  if ((response.headers.get('x-noesis-component-sha256') || '').trim() !== descriptor.sha256) {
    return cancelResponseBody(response, 'depth_bulk_component_header_digest_mismatch');
  }
  if ((response.headers.get('x-noesis-snapshot-id') || '').trim() !== snapshotId) {
    return cancelResponseBody(response, 'depth_bulk_component_snapshot_header_mismatch');
  }

  const output = new Uint8Array(descriptor.byte_count);
  const reader = response.body.getReader();
  let offset = 0;
  try {
    while (true) {
      assertBeforeDeadline(deadlineMs);
      const { done, value } = await reader.read();
      assertBeforeDeadline(deadlineMs);
      if (done) break;
      if (!value || offset + value.byteLength > output.byteLength) {
        throw new Error('depth_bulk_component_length_overflow');
      }
      output.set(value, offset);
      offset += value.byteLength;
    }
  } catch (error) {
    try {
      await reader.cancel(error instanceof Error ? error.message : 'depth_bulk_component_read_failed');
    } catch {
      // Preserve the original stream/validation failure.
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (offset !== output.byteLength) {
    throw new Error('depth_bulk_component_length_truncated');
  }
  assertBeforeDeadline(deadlineMs);
  const digest = sha256Hex(output);
  assertBeforeDeadline(deadlineMs);
  if (digest !== descriptor.sha256) {
    throw new Error('depth_bulk_component_digest_mismatch');
  }
  return output;
};

const reserveRequestAllocation = (requestId: string, byteCount: number): void => {
  const active = controllers.get(requestId);
  if (!active) throw new Error('depth_bulk_request_cancelled');
  if (
    !Number.isSafeInteger(byteCount)
    || byteCount <= 0
    || active.reservedBytes !== 0
    || activeAllocationBytes + byteCount > DEPTH_BULK_MAX_ACTIVE_ALLOCATION_BYTES
  ) {
    throw new Error('depth_bulk_allocation_capacity_exceeded');
  }
  active.reservedBytes = byteCount;
  activeAllocationBytes += byteCount;
};

const loadSnapshot = async (
  requestId: string,
  descriptorValue: unknown,
  expectedCameraId: string,
  baseUrl: string,
  includeRgb: boolean,
  intrinsicsValue: unknown,
  signal: AbortSignal,
  deadlineMs: number,
): Promise<LoadedDepthBulkSnapshot> => {
  const startedAt = performance.now();
  const descriptor = validateDescriptor(descriptorValue);
  if (!expectedCameraId || expectedCameraId.length > 160) {
    throw new Error('depth_bulk_expected_camera_invalid');
  }
  if (typeof includeRgb !== 'boolean') {
    throw new Error('depth_bulk_include_rgb_invalid');
  }
  const intrinsics = normalizeCameraIntrinsics(
    Array.isArray(intrinsicsValue) ? intrinsicsValue.map(Number) : null,
  );
  if (!intrinsics) {
    throw new Error('depth_bulk_intrinsics_missing');
  }
  await validateCaptureEvidenceDigest(descriptor);
  assertBeforeDeadline(deadlineMs);
  const components = descriptor.components;
  for (const component of Object.values(components)) {
    if (!component) continue;
    resolveComponentUrl(
      component,
      expectedCameraId,
      descriptor.snapshot_id,
      descriptor.snapshot_ref,
      descriptor.content_sha256,
      baseUrl,
    );
  }
  const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  if (!littleEndian) {
    throw new Error('depth_bulk_little_endian_runtime_required');
  }
  const filteredDepthBytes = components.depth.byte_count;
  const normalBytes = descriptor.shape[0] * descriptor.shape[1] * 3 * 4;
  const normalSmoothingScratchBytes = descriptor.shape[1] * 3 * 4 * 3;
  const diagnosticsScratchBytes = descriptor.shape[0] * descriptor.shape[1] * 4;
  const requestedTransferBytes = (
    components.depth.byte_count
    + components.conf.byte_count
    + components.mask.byte_count
    + (includeRgb ? components.rgb?.byte_count || 0 : 0)
  );
  reserveRequestAllocation(
    requestId,
    requestedTransferBytes
      + filteredDepthBytes
      + normalBytes
      + normalSmoothingScratchBytes
      + diagnosticsScratchBytes,
  );
  const depthBytes = await fetchComponent(
    components.depth,
    expectedCameraId,
    descriptor.snapshot_id,
    descriptor.snapshot_ref,
    descriptor.content_sha256,
    baseUrl,
    signal,
    deadlineMs,
  );
  const confBytes = await fetchComponent(
    components.conf,
    expectedCameraId,
    descriptor.snapshot_id,
    descriptor.snapshot_ref,
    descriptor.content_sha256,
    baseUrl,
    signal,
    deadlineMs,
  );
  const maskBytes = await fetchComponent(
    components.mask,
    expectedCameraId,
    descriptor.snapshot_id,
    descriptor.snapshot_ref,
    descriptor.content_sha256,
    baseUrl,
    signal,
    deadlineMs,
  );
  const rgbBytes = includeRgb && components.rgb
    ? await fetchComponent(
      components.rgb,
      expectedCameraId,
      descriptor.snapshot_id,
      descriptor.snapshot_ref,
      descriptor.content_sha256,
      baseUrl,
      signal,
      deadlineMs,
    )
    : undefined;

  assertBeforeDeadline(deadlineMs);
  const depth = new Float32Array(depthBytes.buffer, depthBytes.byteOffset, depthBytes.byteLength / 4);
  const conf = new Float32Array(confBytes.buffer, confBytes.byteOffset, confBytes.byteLength / 4);
  const mask = new Uint8Array(maskBytes.buffer, maskBytes.byteOffset, maskBytes.byteLength);
  const normals = deriveCalibratedDepthNormals(
    depth,
    mask,
    descriptor.shape[0],
    descriptor.shape[1],
    intrinsics,
    {
      confidence: conf,
      bilateralRadius: 2,
      sampleRadius: 4,
      normalSmoothingRadius: 1,
    },
  );
  const diagnostics = deriveDepthDiagnostics(depth, conf, mask, descriptor.shape);
  assertBeforeDeadline(deadlineMs);
  const transferBytes = depth.byteLength + conf.byteLength + mask.byteLength + (rgbBytes?.byteLength || 0);
  return {
    ts: descriptor.ts,
    shape: descriptor.shape,
    depth,
    conf,
    mask,
    rgb: rgbBytes,
    rgbShape: rgbBytes
      ? components.rgb?.shape as [number, number, number] | undefined
      : undefined,
    rgbComponentSha256: rgbBytes ? components.rgb?.sha256 : undefined,
    normals,
    normalsShape: [descriptor.shape[0], descriptor.shape[1], 3],
    snapshotId: descriptor.snapshot_id,
    snapshotRef: descriptor.snapshot_ref,
    snapshotContentSha256: descriptor.content_sha256,
    diagnostics,
    transferBytes,
    transferDurationMs: performance.now() - startedAt,
  };
};

self.onmessage = (event: MessageEvent): void => {
  const message = event.data;
  if (!isRecord(message) || typeof message.requestId !== 'string') return;
  const requestId = message.requestId;
  if (message.type === 'cancel') {
    const active = controllers.get(requestId);
    if (active) {
      active.cancelled = true;
      active.controller.abort();
    }
    return;
  }
  if (
    message.type !== 'load'
    || typeof message.baseUrl !== 'string'
    || typeof message.expectedCameraId !== 'string'
    || typeof message.includeRgb !== 'boolean'
    || !Number.isSafeInteger(message.timeoutMs)
    || Number(message.timeoutMs) < 1
    || Number(message.timeoutMs) > DEPTH_BULK_REQUEST_TIMEOUT_MS
  ) return;
  if (controllers.size >= DEPTH_BULK_MAX_ACTIVE_REQUESTS) {
    self.postMessage({
      type: 'error',
      requestId,
      error: 'depth_bulk_request_capacity_exceeded',
    });
    return;
  }
  const controller = new AbortController();
  const timeoutMs = Number(message.timeoutMs);
  const deadlineMs = performance.now() + timeoutMs;
  const active: ActiveRequest = {
    controller,
    cameraId: message.expectedCameraId,
    reservedBytes: 0,
    deadlineMs,
    timeout: setTimeout(() => {
      active.timedOut = true;
      controller.abort();
    }, timeoutMs),
    timedOut: false,
    cancelled: false,
  };
  controllers.set(requestId, active);
  void loadSnapshot(
    requestId,
    message.descriptor,
    message.expectedCameraId,
    message.baseUrl,
    message.includeRgb,
    message.intrinsics,
    controller.signal,
    deadlineMs,
  )
    .then((snapshot) => {
      if (controller.signal.aborted) return;
      const transfer: Transferable[] = [
        snapshot.depth.buffer,
        snapshot.conf.buffer,
        snapshot.mask.buffer,
        snapshot.normals.buffer,
      ];
      if (snapshot.rgb) transfer.push(snapshot.rgb.buffer);
      self.postMessage({ type: 'loaded', requestId, snapshot }, transfer);
    })
    .catch((error: unknown) => {
      if (active.cancelled) return;
      self.postMessage({
        type: 'error',
        requestId,
        error: active.timedOut
          ? 'depth_bulk_transfer_timeout'
          : error instanceof Error ? error.message : String(error),
      });
    })
    .finally(() => {
      clearTimeout(active.timeout);
      activeAllocationBytes = Math.max(
        0,
        activeAllocationBytes - active.reservedBytes,
      );
      if (controllers.get(requestId) === active) controllers.delete(requestId);
    });
};

export {};
