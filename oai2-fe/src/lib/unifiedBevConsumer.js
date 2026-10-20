import {
  createUnifiedBevState, reduceUnifiedBev, advanceUnifiedBev, clearUnifiedBevPeople,
} from './unifiedBevWorld.js';

// Selected authored target in BEV-WC-ACCEPT-R1. Incoming data cannot select it.
export const UNIFIED_BEV_AUTHORED_SCENE_SHA256 = '84bfecdb07904141fcf713d40140a748ec806d432ef89d155b4b8a85ff07d5aa';
const TYPES = new Set(['calibration-bundle', 'floorplan_response', 'tracking', 'world_snapshot', 'world_event']);

/** Called before any asynchronous/diagnostic handlers, with the original envelope. */
export function dispatchUnifiedBevMessage(message, receive, receivedAtMs) {
  if (!TYPES.has(message?.type) || !receive) return false;
  receive(message, receivedAtMs);
  return true;
}

export function createUnifiedBevDisplayState() {
  return Object.freeze({
    world: createUnifiedBevState({ authoredSceneSha256: UNIFIED_BEV_AUTHORED_SCENE_SHA256 }),
    rasters: Object.freeze([]), hasSnapshot: false,
  });
}

function rasterFor(message, map) {
  // Every field is from this exact admitted envelope; a missing optional layer
  // never borrows data from an earlier map or another camera. Maps are bounded
  // by the world consumer (32 cameras, at most 4,194,304 cells per map).
  if (message.scene_prior_diagnostic_meta?.raster_encoding_version !== undefined
    && message.scene_prior_diagnostic_meta.raster_encoding_version !== 1) return null;
  const matches = (v) => typeof v?.grid_b64 === 'string'
    && Array.isArray(v.grid_shape) && v.grid_shape.length === 2
    && v.grid_shape.every((n, i) => n === map.gridShape[i]) && rasterLengthMatches(v, map.gridShape[0] * map.gridShape[1]);
  const compacted = new Map();
  const compact = (v, continuous = false) => {
    if (!matches(v) || (continuous && (v.grid_b64.startsWith('bit:') || v.grid_b64.startsWith('u8:')))) return null;
    if (!compacted.has(v)) compacted.set(v, Object.freeze({ grid_b64: v.grid_b64,
      grid_shape: Object.freeze([...v.grid_shape]), grid_encoding: v.grid_encoding }));
    return compacted.get(v);
  };
  const layer = compact(message.scene_prior_diagnostic_obstacle_height ?? message.scene_prior_diagnostic_height_agl, true);
  const heightLayer = compact(message.scene_prior_diagnostic_height_agl, true);
  const mask = compact(message.scene_prior_diagnostic_observed);
  const rgb = message.scene_prior_diagnostic_surface_rgb;
  const cells = map.gridShape[0] * map.gridShape[1];
  const surfaceRgb = Array.isArray(rgb?.rgb_shape) && rgb.rgb_shape.length === 3 && rgb.rgb_shape[2] === 3
    && rgb.rgb_shape[0] === map.gridShape[0] && rgb.rgb_shape[1] === map.gridShape[1]
    && (rgb.rgb_encoding === undefined || rgb.rgb_encoding === 'rgb-u8-base64')
    && base64LengthMatches(rgb.rgb_b64, cells * 3)
    && rasterLengthMatches({ grid_b64: rgb.observed_b64, grid_encoding: rgb.observed_encoding }, cells)
    ? Object.freeze({ rgb_b64: rgb.rgb_b64, rgb_shape: Object.freeze([...rgb.rgb_shape]),
      rgb_encoding: rgb.rgb_encoding, observed_b64: rgb.observed_b64, observed_encoding: rgb.observed_encoding }) : null;
  if ((!layer || !mask) && (!heightLayer || !mask) && !surfaceRgb) return null;
  return Object.freeze({ cameraId: map.cameraId, mapId: map.mapId,
    snapshotContentSha256: map.snapshotContentSha256, layer, heightLayer, mask, surfaceRgb });
}

function base64LengthMatches(encoded, bytes) {
  if (typeof encoded !== 'string' || encoded.length !== 4 * Math.ceil(bytes / 3)
    || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded)) return false;
  const padding = (3 - bytes % 3) % 3;
  return (encoded.length - encoded.replace(/=+$/, '').length) === padding;
}

function rasterLengthMatches(layer, cells) {
  const encoded = layer.grid_b64;
  if (typeof encoded !== 'string') return false;
  let encoding = 'f32-le-base64'; let prefix = 0; let bytes = cells * 4;
  if (encoded.startsWith('f16:')) { encoding = 'f16-le-base64'; prefix = 4; bytes = cells * 2; }
  else if (encoded.startsWith('u8:')) { encoding = 'u8-base64'; prefix = 3; bytes = cells; }
  else if (encoded.startsWith('bit:')) {
    encoding = 'bitpack-msb-base64'; prefix = encoded.indexOf(':', 4) + 1; bytes = Math.ceil(cells / 8);
    if (prefix <= 4 || Number(encoded.slice(4, prefix - 1)) !== cells) return false;
  }
  return (layer.grid_encoding === undefined || layer.grid_encoding === encoding)
    && base64LengthMatches(encoded.slice(prefix), bytes);
}

/** Owns synchronous ingress and one expiry timer; rendering never renews a clock. */
export function createUnifiedBevConsumer({ onChange = () => {}, now = () => performance.now(),
  schedule = (fn, delay) => setTimeout(fn, delay), cancel = (timer) => clearTimeout(timer) } = {}) {
  let display = createUnifiedBevDisplayState();
  let timer = null;
  const stop = () => { if (timer !== null) cancel(timer); timer = null; };
  const notify = () => { onChange(display); };
  const scheduleExpiry = () => {
    stop();
    const deadlines = [...display.world.people, ...display.world.unavailable].map(p => p.expiresAtMs);
    if (!deadlines.length) return;
    const clock = Math.max(now(), display.world.lastReceivedAtMs ?? 0);
    timer = schedule(() => {
      timer = null;
      const world = advanceUnifiedBev(display.world, Math.max(now(), display.world.lastReceivedAtMs ?? 0));
      display = Object.freeze({ ...display, world });
      notify();
      scheduleExpiry();
    }, Math.max(1, Math.ceil(Math.min(...deadlines) - clock)));
  };
  return Object.freeze({
    getSnapshot: () => display,
    receive(message, receivedAtMs = now()) {
      const world = reduceUnifiedBev(display.world, message, { receivedAtMs });
      const retained = display.rasters.filter(r => world.maps.some(m => (
        m.cameraId === r.cameraId && m.mapId === r.mapId && m.snapshotContentSha256 === r.snapshotContentSha256
      )));
      let rasters = retained.length === display.rasters.length ? display.rasters : retained;
      if (message?.type === 'floorplan_response') {
        rasters = rasters.filter(r => r.cameraId !== message.camera_id);
        if (world.diagnostic.accepted) {
          const map = world.maps.find(m => m.cameraId === message.camera_id);
          const raster = map && rasterFor(message, map);
          if (raster) rasters.push(raster);
        }
      }
      const hasSnapshot = world.bindings !== null && ((world.bindings === display.world.bindings && display.hasSnapshot)
        || (message?.type === 'world_snapshot' && world.diagnostic.accepted));
      display = Object.freeze({ world, rasters: Object.freeze(rasters), hasSnapshot });
      notify();
      scheduleExpiry();
      return display;
    },
    disconnect(reason = 'disconnected') {
      stop();
      display = Object.freeze({ ...display, world: clearUnifiedBevPeople(display.world, reason), hasSnapshot: false });
      notify();
    },
    stop,
  });
}

/** A single display fit over already-projected scene coordinates. No world mapping. */
export function fitUnifiedBevViewport(world, width, height, padding = 30) {
  const points = [...world.maps.flatMap(m => m.sceneCorners),
    ...world.people.flatMap(p => [p.position, ...p.trail.map(t => t.position)])];
  if (!points.length || width <= padding * 2 || height <= padding * 2) return null;
  let minX = Math.min(...points.map(p => p[0])); let maxX = Math.max(...points.map(p => p[0]));
  let minZ = Math.min(...points.map(p => p[2])); let maxZ = Math.max(...points.map(p => p[2]));
  if (maxX - minX < 1) { const x = (minX + maxX) / 2; minX = x - .5; maxX = x + .5; }
  if (maxZ - minZ < 1) { const z = (minZ + maxZ) / 2; minZ = z - .5; maxZ = z + .5; }
  const scale = Math.min((width - padding * 2) / (maxX - minX), (height - padding * 2) / (maxZ - minZ));
  const xOffset = (width - (maxX - minX) * scale) / 2;
  const yOffset = (height - (maxZ - minZ) * scale) / 2;
  return Object.freeze({ minX, maxX, minZ, maxZ, scale,
    point: (p) => [width - (xOffset + (p[0] - minX) * scale), yOffset + (maxZ - p[2]) * scale] });
}

/** Reflect the former label box/anchor; text stays upright and only labels clamp. */
export function unifiedBevLabelPlacement(marker, width, height, textWidth) {
  const boxWidth = textWidth + 10;
  const previousTextX = Math.max(4, Math.min(width - boxWidth - 4, width - marker[0] + 11));
  const textY = Math.max(18, Math.min(height - 4, marker[1] - 9));
  return Object.freeze({ boxLeft: width - (previousTextX - 3 + boxWidth), boxTop: textY - 13,
    boxWidth, boxHeight: 18, textX: width - previousTextX, textY, textAlign: 'right' });
}

/** Original row zero is local max Z: corner3 -> corner2 across, corner3 -> corner0 down. */
export function unifiedBevRasterTransform(map, viewport) {
  const [tl, tr, bl] = [map.sceneCorners[3], map.sceneCorners[2], map.sceneCorners[0]].map(viewport.point);
  const [rows, cols] = map.gridShape;
  return [(tr[0] - tl[0]) / cols, (tr[1] - tl[1]) / cols,
    (bl[0] - tl[0]) / rows, (bl[1] - tl[1]) / rows, tl[0], tl[1]];
}

export function unifiedBevTrailSegments(person, world, config) {
  if (!config.enabled) return [];
  const cutoff = (world.cursor?.publishedAtUs ?? person.observedAtUs) - config.window_s * 1_000_000;
  const points = person.trail.filter(p => p.observedAtUs >= cutoff).slice(-Math.min(128, config.max_points_per_track));
  const segments = []; let segment = [];
  for (const sample of points) {
    if (segment.length && sample.observedAtUs - segment.at(-1).observedAtUs > config.gap_ms * 1000) {
      segments.push(segment); segment = [];
    }
    segment.push(sample);
  }
  if (segment.length) segments.push(segment);
  const stride = Math.max(1, config.draw_stride);
  return segments.map(s => s.filter((_, i) => i % stride === 0 || i === s.length - 1));
}

export function unifiedBevStatus(display, connection) {
  const world = display.world;
  if (connection !== 'open') return connection === 'connecting' ? 'Connecting' : 'Disconnected';
  if (!world.bindings) return 'Scene mapping unavailable';
  if (!world.diagnostic.accepted) return 'Canonical data rejected';
  if (!display.hasSnapshot) return 'Waiting for canonical people';
  if (world.unavailable.length) return `${world.unavailable.length} position${world.unavailable.length === 1 ? '' : 's'} unavailable`;
  if (!world.people.length) return 'No current people';
  return `${world.people.length} ${world.people.length === 1 ? 'person' : 'people'}`;
}
