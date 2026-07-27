import assert from 'node:assert/strict';
import test from 'node:test';

type PostedMessage = Record<string, unknown>;

class FakeWorker {
  static instance: FakeWorker | undefined;

  onmessage: ((event: MessageEvent) => void) | null = null;

  onerror: ((event: ErrorEvent) => void) | null = null;

  messages: PostedMessage[] = [];

  constructor() {
    FakeWorker.instance = this;
  }

  postMessage(message: PostedMessage): void {
    this.messages.push(message);
  }

  terminate(): void {}

  reply(message: PostedMessage): void {
    this.onmessage?.({ data: message } as MessageEvent);
  }
}

Object.assign(globalThis, {
  Worker: FakeWorker,
  window: { location: { origin: 'https://noesis.test' } },
});

const { loadDepthBulkSnapshot } = await import('../src/lib/depthBulkClient');
const loadTestSnapshot = (
  descriptor: unknown,
  cameraId: string,
  options: { deadlineAtMs?: number; includeRgb?: boolean } = {},
) => loadDepthBulkSnapshot(descriptor, cameraId, {
  ...options,
  intrinsics: [100, 100, 0, 0],
});

const snapshot = {
  ts: 1,
  shape: [1, 1] as [number, number],
  depth: new Float32Array([1]),
  conf: new Float32Array([1]),
  mask: new Uint8Array([1]),
  normals: new Float32Array([0, 0, 1]),
  normalsShape: [1, 1, 3] as [number, number, number],
  snapshotId: 'snapshot',
  snapshotRef: 'camera/snapshot.zarr',
  snapshotContentSha256: 'a'.repeat(64),
  diagnostics: {
    median: 1,
    p10: 1,
    p90: 1,
    conf_mean: 1,
    valid_ratio: 1,
    sample_count: 1,
    method: 'client_bulk_exact_snapshot_v1' as const,
  },
  transferBytes: 9,
  transferDurationMs: 1,
};

test('client cancels same-camera work and rejects beyond the global request cap', async () => {
  const first = loadTestSnapshot({}, 'living-room');
  const worker = FakeWorker.instance;
  assert.ok(worker);
  const firstLoad = worker.messages.at(-1);
  assert.equal(firstLoad?.includeRgb, false);

  const replacement = loadTestSnapshot({}, 'living-room');
  await assert.rejects(first, /depth_bulk_request_superseded/);
  assert.equal(worker.messages.at(-2)?.type, 'cancel');
  const replacementLoad = worker.messages.at(-1);

  const kitchen = loadTestSnapshot({}, 'kitchen');
  const kitchenLoad = worker.messages.at(-1);
  const family = loadTestSnapshot({}, 'family-room');
  const familyLoad = worker.messages.at(-1);
  await assert.rejects(
    loadTestSnapshot({}, 'garage'),
    /depth_bulk_request_capacity_exceeded/,
  );

  for (const load of [replacementLoad, kitchenLoad, familyLoad]) {
    worker.reply({
      type: 'loaded',
      requestId: load?.requestId,
      snapshot,
    });
  }
  await Promise.all([replacement, kitchen, family]);
});

test('client inherits an absolute request deadline without renewing it', async () => {
  await assert.rejects(
    loadTestSnapshot({}, 'late-camera', {
      deadlineAtMs: performance.now() - 1,
    }),
    /depth_bulk_request_deadline_exceeded/,
  );

  const deadlineAtMs = performance.now() + 250;
  const pending = loadTestSnapshot({}, 'bounded-camera', { deadlineAtMs });
  const worker = FakeWorker.instance;
  assert.ok(worker);
  const load = worker.messages.at(-1);
  assert.equal(load?.type, 'load');
  assert.equal(typeof load?.timeoutMs, 'number');
  assert.ok(Number(load?.timeoutMs) >= 1);
  assert.ok(Number(load?.timeoutMs) <= 250);
  assert.ok(Number(load?.timeoutMs) < 60_000);
  worker.reply({ type: 'loaded', requestId: load?.requestId, snapshot });
  await pending;
});
