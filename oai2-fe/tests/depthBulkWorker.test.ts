import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import test from 'node:test';

type WorkerReply = {
  type: 'loaded' | 'error';
  requestId: string;
  snapshot?: {
    depth: Float32Array;
    conf: Float32Array;
    mask: Uint8Array;
    rgb?: Uint8Array;
    rgbComponentSha256?: string;
    normals: Float32Array;
    surfaceNormals: Int8Array;
    diagnostics: {
      median: number;
      p10: number;
      p90: number;
      conf_mean: number;
      valid_ratio: number;
      sample_count: number;
      method: string;
    };
    transferBytes: number;
  };
  error?: string;
};

const CAMERA = 'living-room';
const SNAPSHOT_ID = 'snapshot-1';
const SNAPSHOT_REF = 'living-room/20260713/snapshot-1.zarr';
const CONTENT_SHA256 = createHash('sha256').update('snapshot').digest('hex');

const depth = new Float32Array([
  1, 1, 1,
  1, 1, 1,
  1, 1, 1,
]);
const conf = new Float32Array(9).fill(0.9);
const mask = new Uint8Array(9).fill(1);
const rgb = new Uint8Array(27).map((_value, index) => index);
const componentBytes = {
  depth: new Uint8Array(depth.buffer.slice(0)),
  conf: new Uint8Array(conf.buffer.slice(0)),
  mask: new Uint8Array(mask.buffer.slice(0)),
  rgb: new Uint8Array(rgb.buffer.slice(0)),
};

const sha256 = (bytes: Uint8Array): string => (
  createHash('sha256').update(bytes).digest('hex')
);

const componentUrl = (component: keyof typeof componentBytes): string => {
  const query = new URLSearchParams({
    snapshot_ref: SNAPSHOT_REF,
    content_sha256: CONTENT_SHA256,
  });
  return `/api/v1/depth/snapshots/${CAMERA}/${SNAPSHOT_ID}/components/${component}?${query}`;
};

const descriptor = () => ({
  contract: 'noesis.depth.bulk_snapshot',
  contract_version: 1,
  ts: 1_780_000_000_000_000,
  shape: [3, 3],
  snapshot_id: SNAPSHOT_ID,
  snapshot_ref: SNAPSHOT_REF,
  content_sha256: CONTENT_SHA256,
  role: 'capture_event_fused',
  fusion_level: 'intra_capture',
  components: Object.fromEntries(
    (Object.keys(componentBytes) as Array<keyof typeof componentBytes>).map((component) => {
      const bytes = componentBytes[component];
      return [component, {
        component,
        dtype: component === 'mask' || component === 'rgb' ? '|u1' : '<f4',
        shape: component === 'rgb' ? [3, 3, 3] : [3, 3],
        byte_count: bytes.byteLength,
        sha256: sha256(bytes),
        url: componentUrl(component),
      }];
    }),
  ),
  normals: {
    mode: 'client_derived_depth_gradient_v1',
    space: 'camera',
    dtype: 'float32',
  },
});

const workerScope: {
  location: { origin: string };
  crypto: Crypto;
  onmessage?: (event: { data: unknown }) => void;
  postMessage?: (message: WorkerReply, transfer?: Transferable[]) => void;
} = {
  location: { origin: 'https://noesis.test' },
  crypto: webcrypto as unknown as Crypto,
};

Object.assign(globalThis, { self: workerScope });
await import('../src/workers/depthBulkWorker');

const replyWaiters = new Map<string, {
  resolve: (reply: WorkerReply) => void;
  timeout: ReturnType<typeof setTimeout>;
}>();
workerScope.postMessage = (message) => {
  const waiter = replyWaiters.get(message.requestId);
  if (!waiter) return;
  replyWaiters.delete(message.requestId);
  clearTimeout(waiter.timeout);
  waiter.resolve(message);
};

const dispatch = (data: unknown): Promise<WorkerReply> => new Promise((resolve, reject) => {
  const message = typeof data === 'object' && data !== null && (data as { type?: string }).type === 'load'
    ? { includeRgb: false, intrinsics: [100, 100, 1, 1], timeoutMs: 60_000, ...data }
    : data;
  const requestId = (message as { requestId?: unknown })?.requestId;
  if (typeof requestId !== 'string') {
    reject(new Error('test requestId missing'));
    return;
  }
  const timeout = setTimeout(() => {
    replyWaiters.delete(requestId);
    reject(new Error('worker response timed out'));
  }, 2000);
  replyWaiters.set(requestId, { resolve, timeout });
  workerScope.onmessage?.({ data: message });
});

let fetchCount = 0;
let corruptHeader = false;
let corruptBody = false;
let disguisedPartial = false;
let encodedBody = false;
let holdFetch = false;
const heldFetchResolvers: Array<(response: Response) => void> = [];
Object.assign(globalThis, {
  fetch: async (input: URL | RequestInfo, init?: RequestInit) => {
    fetchCount += 1;
    if (holdFetch) {
      return new Promise<Response>((resolve, reject) => {
        heldFetchResolvers.push(resolve);
        const signal = init?.signal;
        const rejectAbort = () => reject(new DOMException('aborted', 'AbortError'));
        if (signal?.aborted) {
          rejectAbort();
        } else {
          signal?.addEventListener('abort', rejectAbort, { once: true });
        }
      });
    }
    const url = new URL(String(input));
    const component = url.pathname.split('/').at(-1) as keyof typeof componentBytes;
    const bytes = componentBytes[component];
    assert.ok(bytes, `unexpected component URL ${url}`);
    const body = bytes.slice();
    if (corruptBody) body[0] ^= 0xff;
    return new Response(body, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(bytes.byteLength),
        'Cache-Control': 'no-store',
        'X-Noesis-Component-Sha256': corruptHeader ? '0'.repeat(64) : sha256(bytes),
        'X-Noesis-Snapshot-Id': SNAPSHOT_ID,
        ...(disguisedPartial ? { 'Content-Range': `bytes 0-${bytes.byteLength - 1}/${bytes.byteLength * 2}` } : {}),
        ...(encodedBody ? { 'Content-Encoding': 'gzip' } : {}),
      },
    });
  },
});

const largeDescriptor = () => {
  const height = 4096;
  const width = 2048;
  const components = Object.fromEntries(
    (['depth', 'conf', 'mask'] as const).map((component) => {
      const byteCount = height * width * (component === 'mask' ? 1 : 4);
      return [component, {
        component,
        dtype: component === 'mask' ? '|u1' : '<f4',
        shape: [height, width],
        byte_count: byteCount,
        sha256: '0'.repeat(64),
        url: componentUrl(component),
      }];
    }),
  );
  return {
    ...descriptor(),
    shape: [height, width],
    components,
  };
};

test('worker streams and verifies an exact same-origin typed-array snapshot', async () => {
  fetchCount = 0;
  corruptHeader = false;
  const reply = await dispatch({
    type: 'load',
    requestId: 'success',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'loaded', reply.error);
  assert.equal(fetchCount, 3);
  assert.deepEqual(Array.from(reply.snapshot?.depth || []), Array.from(depth));
  assert.deepEqual(Array.from(reply.snapshot?.conf || []), Array.from(conf));
  assert.deepEqual(Array.from(reply.snapshot?.mask || []), Array.from(mask));
  assert.equal(reply.snapshot?.transferBytes, 81);
  assert.equal(reply.snapshot?.rgb, undefined);
  assert.equal(reply.snapshot?.normals.length, 27);
  assert.equal(reply.snapshot?.surfaceNormals.length, 27);
  assert.ok(reply.snapshot?.surfaceNormals instanceof Int8Array);
  assert.deepEqual(
    {
      median: reply.snapshot?.diagnostics.median,
      p10: reply.snapshot?.diagnostics.p10,
      p90: reply.snapshot?.diagnostics.p90,
      valid_ratio: reply.snapshot?.diagnostics.valid_ratio,
      sample_count: reply.snapshot?.diagnostics.sample_count,
      method: reply.snapshot?.diagnostics.method,
    },
    {
      median: 1,
      p10: 1,
      p90: 1,
      valid_ratio: 1,
      sample_count: 9,
      method: 'client_bulk_exact_snapshot_v1',
    },
  );
  assert.ok(Math.abs((reply.snapshot?.diagnostics.conf_mean ?? 0) - 0.9) < 1e-6);
});

test('worker verifies and loads snapshots on the LAN HTTP dashboard without WebCrypto', async () => {
  fetchCount = 0;
  const originalOrigin = workerScope.location.origin;
  const originalCrypto = workerScope.crypto;
  workerScope.location.origin = 'http://192.168.3.126:5173';
  workerScope.crypto = {} as Crypto;
  try {
    const reply = await dispatch({
      type: 'load',
      requestId: 'lan-http-no-webcrypto',
      descriptor: descriptor(),
      expectedCameraId: CAMERA,
      baseUrl: workerScope.location.origin,
    });
    assert.equal(reply.type, 'loaded', reply.error);
    assert.equal(fetchCount, 3);
    assert.deepEqual(Array.from(reply.snapshot?.depth || []), Array.from(depth));
    assert.equal(reply.snapshot?.normals.length, 27);
    assert.equal(reply.snapshot?.surfaceNormals.length, 27);
  } finally {
    workerScope.location.origin = originalOrigin;
    workerScope.crypto = originalCrypto;
  }
});

test('worker fetches optional RGB only when explicitly requested', async () => {
  fetchCount = 0;
  const reply = await dispatch({
    type: 'load',
    requestId: 'with-rgb',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
    includeRgb: true,
  });

  assert.equal(reply.type, 'loaded', reply.error);
  assert.equal(fetchCount, 4);
  assert.deepEqual(Array.from(reply.snapshot?.rgb || []), Array.from(rgb));
  assert.equal(reply.snapshot?.rgbComponentSha256, sha256(componentBytes.rgb));
  assert.equal(reply.snapshot?.transferBytes, 108);
});

test('worker rejects aggregate decoded allocations beyond its global budget', async () => {
  holdFetch = true;
  const first = dispatch({
    type: 'load',
    requestId: 'large-allocation-first',
    descriptor: largeDescriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  while (heldFetchResolvers.length === 0) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const second = await dispatch({
    type: 'load',
    requestId: 'large-allocation-second',
    descriptor: largeDescriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  assert.equal(second.type, 'error');
  assert.match(second.error || '', /depth_bulk_allocation_capacity_exceeded/);

  holdFetch = false;
  heldFetchResolvers.shift()?.(new Response(null, { status: 503 }));
  const firstReply = await first;
  assert.equal(firstReply.type, 'error');
  assert.match(firstReply.error || '', /depth_bulk_component_http_503/);
});

test('worker aborts a held fetch at the inherited request deadline', async () => {
  fetchCount = 0;
  holdFetch = true;
  const reply = await dispatch({
    type: 'load',
    requestId: 'inherited-deadline',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
    timeoutMs: 20,
  });
  holdFetch = false;
  heldFetchResolvers.shift();

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_transfer_timeout/);
  assert.equal(fetchCount, 1);
});

test('worker recomputes canonical capture-event evidence before fetching', async () => {
  fetchCount = 0;
  const captureEvent = {
    contract: 'noesis.capture_event_controller',
    parameters: { burst_seconds: 4, depth_agreement_m: 0.18 },
  };
  const canonical = '{"contract":"noesis.capture_event_controller","parameters":{"burst_seconds":4.0,"depth_agreement_m":0.18}}';
  const valid = {
    ...descriptor(),
    capture_event: captureEvent,
    capture_event_evidence_sha256: createHash('sha256').update(canonical).digest('hex'),
  };
  let reply = await dispatch({
    type: 'load',
    requestId: 'capture-evidence-valid',
    descriptor: valid,
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  assert.equal(reply.type, 'loaded', reply.error);
  assert.equal(fetchCount, 3);

  fetchCount = 0;
  reply = await dispatch({
    type: 'load',
    requestId: 'capture-evidence-tampered',
    descriptor: {
      ...valid,
      capture_event: {
        ...captureEvent,
        parameters: { ...captureEvent.parameters, burst_seconds: 5 },
      },
    },
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_capture_evidence_digest_mismatch/);
  assert.equal(fetchCount, 0);
});

test('display normals reject a non-finite center depth', async () => {
  const original = componentBytes.depth.slice();
  new DataView(componentBytes.depth.buffer).setFloat32(4 * 4, Number.NaN, true);
  try {
    const reply = await dispatch({
      type: 'load',
      requestId: 'invalid-normal-center',
      descriptor: descriptor(),
      expectedCameraId: CAMERA,
      baseUrl: workerScope.location.origin,
    });
    assert.equal(reply.type, 'loaded', reply.error);
    assert.deepEqual(Array.from(reply.snapshot?.normals.slice(12, 15) || []), [0, 0, 0]);
  } finally {
    componentBytes.depth.set(original);
  }
});

test('worker rejects inline binary before issuing any component request', async () => {
  fetchCount = 0;
  const invalid = { ...descriptor(), depth_b64: 'forbidden' };
  const reply = await dispatch({
    type: 'load',
    requestId: 'inline-rejected',
    descriptor: invalid,
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_inline_binary_forbidden/);
  assert.equal(fetchCount, 0);
});

test('worker rejects a raw snapshot at the public fused boundary', async () => {
  fetchCount = 0;
  const invalid = { ...descriptor(), role: '', fusion_level: '' };
  const reply = await dispatch({
    type: 'load',
    requestId: 'raw-rejected',
    descriptor: invalid,
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_snapshot_identity_invalid/);
  assert.equal(fetchCount, 0);
});

test('worker binds every component URL to the websocket camera', async () => {
  fetchCount = 0;
  const reply = await dispatch({
    type: 'load',
    requestId: 'camera-mismatch',
    descriptor: descriptor(),
    expectedCameraId: 'kitchen',
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_component_url_camera_mismatch/);
  assert.equal(fetchCount, 0);
});

test('worker rejects a mismatched response digest header', async () => {
  fetchCount = 0;
  corruptHeader = true;
  const reply = await dispatch({
    type: 'load',
    requestId: 'header-mismatch',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_component_header_digest_mismatch/);
  assert.equal(fetchCount, 1);
  corruptHeader = false;
});

test('worker verifies streamed bytes instead of trusting the digest header', async () => {
  fetchCount = 0;
  corruptBody = true;
  const reply = await dispatch({
    type: 'load',
    requestId: 'body-mismatch',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });

  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_component_digest_mismatch/);
  assert.equal(fetchCount, 1);
  corruptBody = false;
});

test('worker rejects partial or encoded component semantics before reading bytes', async () => {
  fetchCount = 0;
  disguisedPartial = true;
  let reply = await dispatch({
    type: 'load',
    requestId: 'partial-semantics',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_component_content_range_invalid/);
  assert.equal(fetchCount, 1);
  disguisedPartial = false;

  fetchCount = 0;
  encodedBody = true;
  reply = await dispatch({
    type: 'load',
    requestId: 'encoded-semantics',
    descriptor: descriptor(),
    expectedCameraId: CAMERA,
    baseUrl: workerScope.location.origin,
  });
  assert.equal(reply.type, 'error');
  assert.match(reply.error || '', /depth_bulk_component_content_encoding_invalid/);
  assert.equal(fetchCount, 1);
  encodedBody = false;
});
