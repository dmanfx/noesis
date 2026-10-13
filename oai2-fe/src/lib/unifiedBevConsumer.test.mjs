import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import {
  createUnifiedBevConsumer, dispatchUnifiedBevMessage, fitUnifiedBevViewport,
  unifiedBevLabelPlacement, unifiedBevRasterTransform, unifiedBevTrailSegments, unifiedBevStatus,
  UNIFIED_BEV_AUTHORED_SCENE_SHA256 as AUTHORED,
} from './unifiedBevConsumer.js';
import { projectUnifiedBevMapPoint } from './unifiedBevWorld.js';

// Synthetic contract fixtures. They are not live or retained walking evidence.
const hash = c => c.repeat(64);
const SOURCE = hash('a'); const MAPPING = hash('b'); const REGISTRATION = hash('c');
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const matrix = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 10, 0, 20, 1];
const frame = { frame_id: 'backend_world_m', revision: 'synthetic-world' };
const bundle = { type: 'calibration-bundle', data: {
  meta: { authored_scene_sha256: AUTHORED, registration_sha256: REGISTRATION },
  cameras: { frame_bindings: { camera: {
    contract: 'noesis.calibration.frame_binding', contract_version: 1,
    calibration_frame: { frame_id: 'backend_world_m', revision: 'synthetic-calibration' }, world_frame: frame,
    transform_sha256: SOURCE, target_from_calibration_col_major: [...identity.slice(0, 12), 1000, 0, 1000, 1],
  } } },
  world_frame_presentations: { [frame.revision]: {
    contract: 'noesis.scene_prior.world_to_scene_presentation', contract_version: 1,
    accepted: true, render_frame: 'menon_scene', world_frame: frame,
    authored_scene_sha256: AUTHORED, registration_sha256: REGISTRATION, provenance: { sha256: REGISTRATION },
    world_to_scene_col_major: matrix, world_to_scene_sha256: MAPPING,
    scene_revision_id: hash('d'), source_transform_sha256s: [SOURCE],
  } },
} };
const map = { type: 'floorplan_response', camera_id: 'camera', frame: 'camera_local_ground_m', units: 'meters',
  world_frame: 'backend_world_m', world_frame_revision: frame.revision, scene_prior_only: true, display_source: 'pcf',
  ts: 100, snapshot_id: 'synthetic-map', snapshot_content_sha256: hash('e'),
  bounds: { min_x: 0, max_x: 2, min_z: 0, max_z: 2 },
  scene_prior_meta: { contract: 'noesis.scene_prior.floorplan_composite', contract_version: 1,
    source_frame: 'backend_world_m', target_frame: 'camera_local_ground_m', source_frame_revision: frame.revision,
    frame_transform_sha256: SOURCE, camera_geometry_source: 'current_calibrated_extrinsics', prior_id: 'synthetic-map',
    revision_manifest_sha256: hash('e'), floor_y_m: 0,
    raster_orientation: 'row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x' },
  scene_prior_diagnostic_meta: { contract: 'noesis.scene_prior.diagnostic_layers', contract_version: 1,
    raster_orientation: 'row_zero_max_z_rows_toward_min_z_columns_min_x_to_max_x',
    bounds: { min_x: 0, max_x: 2, min_z: 0, max_z: 2 }, grid_shape: [2, 2], resolution_m: 1,
    camera_position_world_m: [5, 0, 6], camera_right_world_xz: [1, 0], camera_forward_world_xz: [0, 1] },
  scene_prior_diagnostic_height_agl: { grid_shape: [2, 2], grid_b64: Buffer.from(new Float32Array([0, 1, 2, 3]).buffer).toString('base64') },
  scene_prior_diagnostic_observed: { grid_shape: [2, 2], grid_b64: 'bit:4:8A==', grid_encoding: 'bitpack-msb-base64' },
};
const clone = value => structuredClone(value);
const producer = { runtime: 'test', instance_id: 'consumer-fixture', run_id: 'synthetic-run', software_revision: 'r1' };
function entity(time = 1_000_000, changes = {}) {
  const position = { x: 6, y: 0, z: 7 };
  const covariance = { values: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
  return { entity_id: 'visitor:synthetic:7:g1', subject: { subject_id: 'visitor:synthetic:7:g1', kind: 'visitor', stable_id: 7, generation: 1 },
    lifecycle: 'present', position_state: 'measured', observed_at_us: time, stale_after_us: time + 2_000_000,
    position, covariance, world_frame: 'backend_world_m', world_frame_revision: frame.revision,
    world_transform_sha256: SOURCE, calibration_revision: 'synthetic-calibration', position_quantity: 'ground_footprint',
    continuity_segment_id: 1, room_id: null,
    sources: [{ observation_id: `observation:${time}`, camera_id: 'camera', observed_at_us: time, position, covariance,
      accepted: true, world_frame: 'backend_world_m', world_frame_revision: frame.revision,
      world_transform_sha256: SOURCE, calibration_revision: 'synthetic-calibration', position_state: 'measured' }], ...changes };
}
function snapshot(sequence = 1, entities = [entity()], changes = {}) {
  const time = 1_000_000 + sequence * 10_000;
  return { contract: 'noesis.world.snapshot', contract_version: 1, snapshot_id: `synthetic-run:${sequence}`, producer,
    sequence, observed_start_us: Math.min(time, ...entities.map(e => e.observed_at_us)),
    observed_end_us: Math.max(time, ...entities.map(e => e.observed_at_us)), published_at_us: time,
    frame: 'backend_world_m', units: 'meters', entities, ...changes };
}
function envelope(payload, type = 'world_snapshot') {
  const cohort = { source_id: 0, frame_id: payload.sequence, observed_at_us: payload.published_at_us, tracking_publication_sequence: payload.sequence };
  return { type, payload, cohort, ...cohort };
}
function tracking(payload, events = []) {
  const { payload: _, ...wire } = envelope(payload);
  return { ...wire, type: 'tracking', source_epoch: 0, world_snapshot: payload, world_events: events };
}
function harness(initialClock = 1000) {
  let clock = initialClock; let nextId = 1;
  const timers = new Map(); const history = [];
  const consumer = createUnifiedBevConsumer({ onChange: s => history.push(s), now: () => clock,
    schedule: (fn, delay) => { const id = nextId++; timers.set(id, { fn, delay }); return id; }, cancel: id => timers.delete(id) });
  consumer.receive(clone(bundle), clock);
  return { consumer, timers, history, tick(ms) { clock = ms; const timer = [...timers.values()][0]; timers.clear(); timer?.fn(); } };
}
const config = { enabled: true, window_s: 8, max_points_per_track: 128, draw_stride: 1, gap_ms: 650 };
function feed(consumer, payload, clock) {
  consumer.receive(tracking(payload), clock);
  return consumer.receive(envelope(payload), clock);
}

test('original exact envelopes dispatch synchronously in order; diagnostic camera BEV never enters', () => {
  const h = harness(); const s = snapshot(); const first = tracking(s);
  const order = [];
  const receive = (message, clock) => { order.push(message); h.consumer.receive(message, clock); };
  dispatchUnifiedBevMessage(first, receive, 1000);
  assert.equal(order[0], first); assert.equal(h.consumer.getSnapshot().world.people.length, 0);
  const standalone = envelope(s); dispatchUnifiedBevMessage(standalone, receive, 1000);
  assert.equal(order[1], standalone); assert.deepEqual(h.consumer.getSnapshot().world.people[0].position, [22, 0, 34]);
  assert.equal(h.consumer.getSnapshot().world.people[0].sceneTransformCount, 1);
  assert.equal(dispatchUnifiedBevMessage({ type: 'bev-frame', footpoints: [{ stableId: 99, x: 1, y: 1 }] }, receive, 1000), false);
  assert.equal(order.length, 2);
});

test('mismatched nested snapshot and missing strict anchors are rejected without making people', () => {
  const h = harness(); const s = snapshot(); h.consumer.receive(tracking(s), 1000);
  const mismatch = clone(s); mismatch.entities[0].position.x += 1;
  const state = h.consumer.receive(envelope(mismatch), 1000);
  assert.equal(state.world.diagnostic.reason, 'snapshot_mismatch'); assert.equal(state.world.people.length, 0);
  const broken = tracking(snapshot(2)); delete broken.world_events;
  assert.equal(h.consumer.receive(broken, 1000).world.diagnostic.accepted, false);
});

test('exact standalone events are ordered diagnostics and cannot create or move a marker', () => {
  const h = harness(); const s = snapshot(); const e = s.entities[0];
  const event = { contract: 'noesis.world.event', contract_version: 1, event_id: 'synthetic-event:1', producer,
    sequence: 1, event_type: 'appeared', entity_id: e.entity_id, subject: e.subject,
    observed_at_us: e.observed_at_us, published_at_us: s.published_at_us, frame: 'backend_world_m', units: 'meters' };
  const wire = { ...envelope(s), type: 'world_event', payload: event };
  assert.equal(h.consumer.receive(wire, 1000).world.diagnostic.reason, 'event_without_exact_snapshot');
  h.consumer.receive(tracking(s, [event]), 1000); h.consumer.receive(envelope(s), 1000);
  const point = h.consumer.getSnapshot().world.people[0].position;
  assert.equal(h.consumer.receive(wire, 1000).world.diagnostic.accepted, true);
  assert.deepEqual(h.consumer.getSnapshot().world.people[0].position, point);
  assert.equal(h.consumer.receive(wire, 1000).world.diagnostic.accepted, false);
  assert.equal(h.consumer.getSnapshot().world.people.length, 1);
});

test('standalone current membership draws one identity and applies canonical empty/lost immediately', () => {
  const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
  assert.equal(h.consumer.getSnapshot().hasSnapshot, true);
  assert.equal(unifiedBevStatus(h.consumer.getSnapshot(), 'open'), '1 person');
  feed(h.consumer, snapshot(2, []), 1010);
  assert.equal(unifiedBevStatus(h.consumer.getSnapshot(), 'open'), 'No current people');
  assert.equal(h.consumer.getSnapshot().world.people.length, 0);
  feed(h.consumer, snapshot(3), 1020);
  feed(h.consumer, snapshot(4, [entity(1_000_000, { lifecycle: 'lost' })]), 1030);
  assert.equal(h.consumer.getSnapshot().world.people.length, 0);
});

test('expiry runs without messages; repeated held samples do not renew freshness or append motion', () => {
  const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
  const before = h.consumer.getSnapshot().world.people[0];
  const held = entity(1_000_000, { lifecycle: 'held', position_state: 'held' });
  feed(h.consumer, snapshot(2, [held]), 1500);
  const after = h.consumer.getSnapshot().world.people[0];
  assert.equal(after.expiresAtMs, before.expiresAtMs); assert.deepEqual(after.trail, before.trail);
  assert.equal(h.timers.size, 1);
  h.tick(before.expiresAtMs);
  assert.equal(h.consumer.getSnapshot().world.people.length, 0); assert.equal(h.timers.size, 0);
});

test('disconnect clears people/trails/pending immediately and reconnect retains replay fences', () => {
  const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
  h.consumer.receive(tracking(snapshot(2)), 1001); h.consumer.disconnect();
  const cleared = h.consumer.getSnapshot();
  assert.equal(cleared.world.people.length, 0); assert.equal(cleared.world.pending.length, 0);
  assert.equal(cleared.hasSnapshot, false); assert.equal(h.timers.size, 0);
  assert.equal(unifiedBevStatus(cleared, 'open'), 'Waiting for canonical people');
  assert.equal(h.consumer.receive(envelope(snapshot()), 1002).world.diagnostic.accepted, false);
});

test('run, identity generation, continuity and mapping boundaries cannot join trails', () => {
  for (const change of [e => { e.continuity_segment_id = 2; }, e => { e.subject.generation = 2; }]) {
    const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
    const e = entity(1_020_000); change(e);
    const person = feed(h.consumer, snapshot(2, [e]), 1020).world.people[0];
    assert.equal(person.trail.length, 1); assert.equal(person.trailBreak, true);
  }
  const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
  const newRun = snapshot(1, [entity(1_030_000)], { producer: { ...producer, run_id: 'new-run' }, published_at_us: 1_040_000, observed_end_us: 1_030_000 });
  assert.equal(feed(h.consumer, newRun, 1040).world.people[0].trail.length, 1);
  const changed = clone(bundle); changed.data.world_frame_presentations[frame.revision].world_to_scene_sha256 = hash('f');
  const cleared = h.consumer.receive(changed, 1041);
  assert.equal(cleared.world.people.length, 0); assert.equal(cleared.hasSnapshot, false);
});

test('a wrong world revision becomes unavailable; a rejected calibration clears all geometry and people', () => {
  const h = harness(); h.consumer.receive(clone(map), 1000);
  const e = entity(); e.world_frame_revision = 'wrong';
  const refused = h.consumer.receive(envelope(snapshot(1, [e])), 1000);
  assert.equal(refused.world.people.length, 0); assert.equal(refused.world.unavailable.length, 1);
  assert.equal(unifiedBevStatus(refused, 'open'), '1 position unavailable');
  const bad = clone(bundle); bad.data.meta.authored_scene_sha256 = hash('f');
  const cleared = h.consumer.receive(bad, 1001);
  assert.equal(cleared.world.bindings, null); assert.equal(cleared.world.maps.length, 0); assert.equal(cleared.rasters.length, 0);
});

test('only exact admitted map identity retains a bounded raster; tracking does not redecode it', () => {
  const h = harness(); const admitted = h.consumer.receive(clone(map), 1000);
  assert.equal(admitted.rasters.length, 1); const rasters = admitted.rasters;
  h.consumer.receive(envelope(snapshot()), 1000);
  assert.equal(h.consumer.getSnapshot().rasters, rasters);
  const bad = clone(map); bad.snapshot_content_sha256 = hash('f');
  h.consumer.receive(bad, 1001);
  assert.equal(h.consumer.getSnapshot().world.maps.length, 0); assert.equal(h.consumer.getSnapshot().rasters.length, 0);
  const invalidRaster = clone(map); invalidRaster.scene_prior_diagnostic_height_agl.grid_encoding = 'f16-le-base64';
  const noImage = h.consumer.receive(invalidRaster, 1002);
  assert.equal(noImage.world.maps.length, 1); assert.equal(noImage.rasters.length, 0);
});

test('fitted person and map use the same scene plane; raster row orientation is preserved exactly', () => {
  const h = harness(); h.consumer.receive(clone(map), 1000); h.consumer.receive(envelope(snapshot()), 1000);
  const { world } = h.consumer.getSnapshot(); const viewport = fitUnifiedBevViewport(world, 500, 300);
  assert.deepEqual(viewport.point(world.people[0].position), [250, 150]);
  const m = world.maps[0]; const [a, b, c, d, x, y] = unifiedBevRasterTransform(m, viewport);
  const imagePoint = (u, v) => [a * u + c * v + x, b * u + d * v + y];
  assert.deepEqual(imagePoint(0, 0), viewport.point(m.sceneCorners[3]));
  assert.deepEqual(imagePoint(2, 0), viewport.point(m.sceneCorners[2]));
  assert.deepEqual(imagePoint(0, 2), viewport.point(m.sceneCorners[0]));
  assert.deepEqual(imagePoint(1, 1), viewport.point(world.people[0].position));
});

test('horizontal display reflection preserves asymmetric points, fit and original data', () => {
  const h = harness(); h.consumer.receive(clone(map), 1000); h.consumer.receive(envelope(snapshot()), 1000);
  const { world } = h.consumer.getSnapshot(); const before = JSON.stringify(world);
  const viewport = fitUnifiedBevViewport(world, 500, 300);
  assert.deepEqual([viewport.minX, viewport.maxX, viewport.minZ, viewport.maxZ, viewport.scale], [20, 24, 32, 36, 60]);
  assert.deepEqual(viewport.point([20.5, 0, 33.25]), [340, 195], 'formerly left160/195 is now right340/195');
  assert.deepEqual(viewport.point([23.5, 0, 35.75]), [160, 45], 'formerly right340/45 is now left160/45');
  for (const padding of [15, 30, 45]) {
    const fit = fitUnifiedBevViewport(world, 500, 300, padding);
    const previousPoint = p => [(500 - (fit.maxX - fit.minX) * fit.scale) / 2 + (p[0] - fit.minX) * fit.scale,
      (300 - (fit.maxZ - fit.minZ) * fit.scale) / 2 + (fit.maxZ - p[2]) * fit.scale];
    for (const p of [...world.maps[0].sceneCorners, world.people[0].position]) {
      const previous = previousPoint(p);
      assert.deepEqual(fit.point(p), [500 - previous[0], previous[1]]);
    }
  }
  assert.equal(JSON.stringify(world), before, 'display calls do not alter canonical coordinates or provenance');
});

test('horizontal reflection shares rotated/skewed admitted raster, marker, trail and bounds geometry', () => {
  const h = harness(); const rotatedBundle = clone(bundle); const rotatedMap = clone(map);
  const theta = Math.PI / 5; const phi = Math.PI / 6;
  const ct = Math.cos(theta); const st = Math.sin(theta); const cp = Math.cos(phi); const sp = Math.sin(phi);
  // A proper 3D similarity, not a fabricated 2D shear; its ground projection is skewed.
  rotatedBundle.data.world_frame_presentations[frame.revision].world_to_scene_col_major = [
    2 * cp, 2 * ct * sp, 2 * st * sp, 0, -2 * sp, 2 * ct * cp, 2 * st * cp, 0,
    0, -2 * st, 2 * ct, 0, 10, 0, 20, 1,
  ];
  rotatedMap.scene_prior_diagnostic_meta.camera_right_world_xz = [.6, .8];
  rotatedMap.scene_prior_diagnostic_meta.camera_forward_world_xz = [-.8, .6];
  h.consumer.receive(rotatedBundle, 1000);
  const admitted = h.consumer.receive(rotatedMap, 1000);
  assert.equal(admitted.world.diagnostic.accepted, true); assert.equal(admitted.rasters.length, 1);
  const worldPosition = ([x, z]) => ({ x: 5 + .6 * x - .8 * z, y: 0, z: 6 + .8 * x + .6 * z });
  h.consumer.receive(envelope(snapshot(1, [entity(1_000_000, { position: worldPosition([.5, .5]) })])), 1000);
  const { world } = feed(h.consumer, snapshot(2, [entity(1_020_000, { position: worldPosition([1.5, 1.25]) })]), 1020);
  assert.equal(world.diagnostic.accepted, true); assert.equal(world.people[0].trail.length, 2);
  const before = JSON.stringify(world); const viewport = fitUnifiedBevViewport(world, 640, 360);
  const m = world.maps[0]; const [a, b, c, d, x, y] = unifiedBevRasterTransform(m, viewport);
  const imagePoint = (u, v) => [a * u + c * v + x, b * u + d * v + y];
  const close = (actual, expected) => actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-9));
  const previousPoint = p => [(640 - (viewport.maxX - viewport.minX) * viewport.scale) / 2 + (p[0] - viewport.minX) * viewport.scale,
    (360 - (viewport.maxZ - viewport.minZ) * viewport.scale) / 2 + (viewport.maxZ - p[2]) * viewport.scale];
  for (const [u, v] of [[0, 0], [2, 0], [0, 2], [2, 2], [.5, .5], [1.5, .5], [.5, 1.5], [1.5, 1.5]]) {
    const scene = projectUnifiedBevMapPoint(m, [u, 2 - v]); const previous = previousPoint(scene);
    close(imagePoint(u, v), [640 - previous[0], previous[1]]);
    close(imagePoint(u, v), viewport.point(scene));
  }
  assert.ok(Math.abs(a * c + b * d) > 1e-3, 'test exercises a skewed projected raster basis');
  const person = world.people[0];
  for (const p of [...m.sceneCorners, person.position, ...unifiedBevTrailSegments(person, world, config)[0].map(t => t.position)]) {
    const previous = previousPoint(p); close(viewport.point(p), [640 - previous[0], previous[1]]);
  }
  close(person.position, projectUnifiedBevMapPoint(m, [1.5, 1.25]));
  close(person.trail[0].position, projectUnifiedBevMapPoint(m, [.5, .5]));
  assert.equal(person.sceneTransformCount, 1);
  assert.equal(JSON.stringify(world), before, 'raster cells, matrices, identities and trail samples stay original');
});

test('reflected label boxes and anchors clamp labels only with upright right-aligned text', () => {
  for (const marker of [[390, 80], [530, 150], [10, 12], [250, 320]]) {
    const before = [...marker]; const placed = unifiedBevLabelPlacement(marker, 500, 300, 90);
    const oldTextX = Math.max(4, Math.min(396, 500 - marker[0] + 11));
    const oldTextY = Math.max(18, Math.min(296, marker[1] - 9));
    assert.deepEqual(placed, { boxLeft: 500 - (oldTextX - 3 + 100), boxTop: oldTextY - 13,
      boxWidth: 100, boxHeight: 18, textX: 500 - oldTextX, textY: oldTextY, textAlign: 'right' });
    assert.ok(placed.boxLeft >= 1 && placed.boxLeft + placed.boxWidth <= 499);
    assert.deepEqual(marker, before, 'no marker edge clamping');
  }
  assert.equal(unifiedBevLabelPlacement([390, 80], 500, 300, 90).boxLeft, 282);
  assert.equal(unifiedBevLabelPlacement([390, 80], 500, 300, 90).textX, 379);
});

test('relevant trail settings affect only measured history, without fabricating a held/predicted endpoint', () => {
  const h = harness(); h.consumer.receive(envelope(snapshot()), 1000);
  feed(h.consumer, snapshot(2, [entity(1_020_000, { position: { x: 7, y: 0, z: 7 } })]), 1020);
  const held = entity(1_020_000, { position_state: 'held', position: { x: 9, y: 0, z: 7 } });
  const { world } = feed(h.consumer, snapshot(3, [held]), 1030); const p = world.people[0];
  const trail = unifiedBevTrailSegments(p, world, config);
  assert.equal(trail[0].length, 2); assert.notDeepEqual(trail[0].at(-1).position, p.position);
  assert.deepEqual(unifiedBevTrailSegments(p, world, { ...config, enabled: false }), []);
  assert.equal(unifiedBevTrailSegments(p, world, { ...config, max_points_per_track: 1 })[0].length, 1);
  assert.deepEqual(unifiedBevTrailSegments(p, world, { ...config, window_s: .001 }), []);
});

test('actual WebSocket hook dispatches canonical messages before existing callbacks and clears on errors/close', async () => {
  const context = { cleanup: [], sockets: [], trace: [] };
  globalThis.__unifiedBevHookTest = context;
  const previousWebSocket = globalThis.WebSocket;
  class FakeSocket { static OPEN = 1; constructor() { this.readyState = 0; context.sockets.push(this); } send() {} close() {} }
  globalThis.WebSocket = FakeSocket;
  try {
    const result = await build({ entryPoints: [new URL('../hooks/useWebSocketClient.ts', import.meta.url).pathname],
      bundle: true, write: false, format: 'esm', platform: 'node', plugins: [{ name: 'bounded-hook-test', setup(builder) {
        builder.onResolve({ filter: /^(react|\.\.\/lib\/(calibration|camera|wsLogger))$/ }, args => ({ path: args.path, namespace: 'test-stub' }));
        builder.onLoad({ filter: /.*/, namespace: 'test-stub' }, args => ({ contents: args.path === 'react'
          ? 'export const useRef=v=>({current:v}); export const useState=v=>[typeof v==="function"?v():v,()=>{}]; export const useEffect=fn=>globalThis.__unifiedBevHookTest.cleanup.push(fn());'
          : args.path.endsWith('calibration') ? 'export const setCalibration=v=>globalThis.__unifiedBevHookTest.trace.push(["normalize",v]);'
            : args.path.endsWith('camera') ? 'export const detectCameraKey=()=>null;'
              : 'export const wsLog=new Proxy({}, {get:()=>()=>{}});' }));
      } }] });
    const { useWebSocketClient } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
    const h = harness(0); const canonical = [];
    useWebSocketClient('ws://synthetic', { onStats() {}, onUnifiedBevMessage(message, receipt) {
      canonical.push(message); context.trace.push(['canonical', message]); h.consumer.receive(message, receipt);
    }, onCalibration: b => context.trace.push(['old-calibration', b]), onFloorplan: f => context.trace.push(['old-floorplan', f]),
    onBevMeta: b => context.trace.push(['old-bev', b]), onDisconnect: h.consumer.disconnect });
    const socket = context.sockets[0];
    socket.onmessage({ data: JSON.stringify(bundle) });
    assert.deepEqual(context.trace.map(v => v[0]), ['canonical', 'normalize', 'old-calibration']);
    const s = snapshot(); socket.onmessage({ data: JSON.stringify(tracking(s)) });
    assert.equal(h.consumer.getSnapshot().world.people.length, 0);
    socket.onmessage({ data: JSON.stringify(envelope(s)) });
    assert.equal(h.consumer.getSnapshot().world.people.length, 1);
    assert.deepEqual(canonical.map(v => v.type), ['calibration-bundle', 'tracking', 'world_snapshot']);
    socket.onmessage({ data: JSON.stringify(map) }); socket.onmessage({ data: JSON.stringify({ type: 'bev-frame' }) });
    assert.equal(canonical.at(-1).type, 'floorplan_response'); assert.equal(context.trace.at(-1)[0], 'old-bev');
    socket.onerror({}); assert.equal(h.consumer.getSnapshot().world.people.length, 0);
    socket.onclose({ code: 1000, reason: 'synthetic' }); assert.equal(h.consumer.getSnapshot().hasSnapshot, false);
    context.cleanup.forEach(fn => fn?.());
  } finally { globalThis.WebSocket = previousWebSocket; delete globalThis.__unifiedBevHookTest; }
});

test('actual canvas component draws the admitted raster and exact canonical marker in one fitted space', async () => {
  const h = harness(); const partialMap = clone(map);
  partialMap.scene_prior_diagnostic_observed.grid_b64 = 'bit:4:oA==';
  h.consumer.receive(partialMap, 1000);
  h.consumer.receive(envelope(snapshot(1, [entity(1_000_000, { position: { x: 6.5, y: 0, z: 7.25 } })])), 1000);
  const calls = []; const cleanup = []; const rasterPixels = [];
  let drawingMatrix = [1, 0, 0, 1, 0, 0]; const matrixStack = [];
  const ctx = new Proxy({
    measureText: label => ({ width: label.length * 6 }),
    setTransform(...m) { drawingMatrix = m; calls.push(['setTransform', ...m]); },
    save() { matrixStack.push([...drawingMatrix]); calls.push(['save']); },
    restore() { drawingMatrix = matrixStack.pop(); calls.push(['restore']); },
    transform(...m) {
      const [a, b, c, d, e, f] = drawingMatrix; const [u, v, w, x, y, z] = m;
      drawingMatrix = [a * u + c * v, b * u + d * v, a * w + c * x, b * w + d * x, a * y + c * z + e, b * y + d * z + f];
      calls.push(['transform', ...m]);
    },
    fillText(label, x, y) { calls.push(['fillText', label, x, y, { textAlign: this.textAlign, matrix: [...drawingMatrix] }]); },
  }, {
    get(target, key) { return target[key] ?? ((...args) => calls.push([key, ...args])); },
  });
  const canvas = { getBoundingClientRect: () => ({ width: 500, height: 300 }), getContext: () => ctx };
  const previous = { document: globalThis.document, window: globalThis.window, ResizeObserver: globalThis.ResizeObserver };
  globalThis.__unifiedBevCanvasTest = { canvas, cleanup };
  globalThis.document = { createElement: () => ({ getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData(image) { rasterPixels.push(Array.from(image.data)); },
  }) }) };
  globalThis.window = { devicePixelRatio: 1 };
  globalThis.ResizeObserver = class { observe() {} disconnect() {} };
  try {
    const result = await build({ entryPoints: [new URL('../components/UnifiedBevView.tsx', import.meta.url).pathname],
      bundle: true, write: false, format: 'esm', platform: 'node', jsx: 'transform',
      jsxFactory: 'React.createElement', jsxFragment: 'React.Fragment', plugins: [{ name: 'bounded-canvas-test', setup(builder) {
        builder.onResolve({ filter: /^react(?:\/jsx-runtime)?$/ }, args => ({ path: args.path, namespace: 'canvas-stub' }));
        builder.onLoad({ filter: /.*/, namespace: 'canvas-stub' }, args => ({ contents: args.path === 'react/jsx-runtime'
          ? 'export const jsx=(tag,props,key)=>({tag,props,key}); export const jsxs=jsx; export const Fragment="fragment";'
          : 'const React={createElement:(tag,props,...children)=>({tag,props,children})}; export default React; export const useRef=()=>({current:globalThis.__unifiedBevCanvasTest.canvas}); export const useMemo=fn=>fn(); export const useEffect=fn=>globalThis.__unifiedBevCanvasTest.cleanup.push(fn());' }));
      } }] });
    const { UnifiedBevView } = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles[0].text).toString('base64')}`);
    const before = JSON.stringify(h.consumer.getSnapshot());
    const view = UnifiedBevView({ display: h.consumer.getSnapshot(), connection: 'open', trailEnabled: true });
    assert.equal(view.props['data-person-count'], 1);
    assert.deepEqual(calls.find(c => c[0] === 'arc').slice(1, 4), [190, 120, 6], 'off-center marker reflects former310/120');
    assert.deepEqual(calls.find(c => c[0] === 'fillRect').slice(1), [124, 98, 58, 18]);
    assert.deepEqual(calls.find(c => c[0] === 'fillText').slice(1), ['Person 7', 179, 111,
      { textAlign: 'right', matrix: [1, 0, 0, 1, 0, 0] }], 'reflected label anchor with positive upright text basis');
    const findPerson = node => {
      if (!node || typeof node !== 'object') return null;
      if (node.tag === 'li') return node;
      const children = node.props?.children ?? node.children ?? [];
      for (const child of [children].flat(Infinity)) { const found = findPerson(child); if (found) return found; }
      return null;
    };
    const personNode = findPerson(view);
    assert.equal(personNode.props['data-scene-position'], '[23,0,34.5]');
    assert.equal(personNode.props['data-world-position'], '[6.5,0,7.25]');
    assert.equal(personNode.props['data-cohort'], JSON.stringify(h.consumer.getSnapshot().world.people[0].cohort));
    assert.equal(personNode.props['data-scene-transform-count'], 1);
    assert.equal(JSON.stringify(h.consumer.getSnapshot()), before, 'drawing leaves complete source state unchanged');
    assert.equal(calls.filter(c => c[0] === 'drawImage').length, 1);
    assert.equal(calls.filter(c => c[0] === 'closePath').length, 0, 'diagnostic bounds are hidden by default');
    assert.ok(rasterPixels[0].slice(0, 3).every(v => v >= 110), 'observed floor pixels have readable contrast');
    assert.equal(rasterPixels[0][3], 255, 'observed cells are fully opaque');
    assert.deepEqual(rasterPixels[0].slice(4, 8), [0, 0, 0, 0], 'unobserved cells remain transparent');
    const { world } = h.consumer.getSnapshot(); const viewport = fitUnifiedBevViewport(world, 500, 300);
    assert.deepEqual(calls.find(c => c[0] === 'transform').slice(1), unifiedBevRasterTransform(world.maps[0], viewport));
    assert.equal(world.people[0].sceneTransformCount, 1);
    calls.length = 0;
    UnifiedBevView({ display: h.consumer.getSnapshot(), connection: 'open', trailEnabled: true, debug: true });
    assert.equal(calls.filter(c => c[0] === 'closePath').length, 1, 'explicit debug state restores exact map bounds');
    assert.deepEqual(calls.find(c => c[0] === 'arc').slice(1, 4), [190, 120, 6]);
    assert.deepEqual(calls.find(c => c[0] === 'moveTo').slice(1), viewport.point(world.maps[0].sceneCorners[0]));
    assert.deepEqual(calls.filter(c => c[0] === 'lineTo').map(c => c.slice(1)), world.maps[0].sceneCorners.slice(1).map(viewport.point));
    feed(h.consumer, snapshot(2, [entity(1_020_000, { position: { x: 5.5, y: 0, z: 6.5 } })]), 1020); calls.length = 0;
    const trailBefore = JSON.stringify(h.consumer.getSnapshot());
    UnifiedBevView({ display: h.consumer.getSnapshot(), connection: 'open', trailEnabled: true });
    assert.deepEqual(calls.find(c => c[0] === 'moveTo').slice(1), [190, 120]);
    assert.deepEqual(calls.find(c => c[0] === 'lineTo').slice(1), [310, 210]);
    assert.deepEqual(calls.find(c => c[0] === 'arc').slice(1, 4), [310, 210, 6]);
    assert.equal(JSON.stringify(h.consumer.getSnapshot()), trailBefore);
    feed(h.consumer, snapshot(3, []), 1030); calls.length = 0;
    const emptyView = UnifiedBevView({ display: h.consumer.getSnapshot(), connection: 'open', trailEnabled: true });
    assert.equal(emptyView.props['data-person-count'], 0);
    assert.equal(calls.filter(c => c[0] === 'arc').length, 0);
    assert.equal(calls.filter(c => c[0] === 'clearRect').length, 1);
    cleanup.forEach(fn => fn?.());
  } finally { Object.assign(globalThis, previous); delete globalThis.__unifiedBevCanvasTest; }
});
