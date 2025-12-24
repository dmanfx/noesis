import { CameraKey } from './camera';

type Point = { x: number; y: number };
type Trails = Record<CameraKey, Record<string, { points: Point[]; colorId: number }>>;

const MAX_POINTS = 300;

export class TrailStore {
  trails: Trails = { 'living-room': {}, 'kitchen': {}, 'family-room': {} };

  push(cam: CameraKey, key: string, pt: Point, colorId: number) {
    if (!this.trails[cam][key]) this.trails[cam][key] = { points: [], colorId };
    const entry = this.trails[cam][key];
    entry.colorId = colorId;
    const arr = entry.points;
    arr.push(pt);
    if (arr.length > MAX_POINTS) arr.shift();
  }

  // Insert a gap marker so drawTrails breaks the polyline for this track
  pushBreak(cam: CameraKey, key: string, colorId: number) {
    if (!this.trails[cam][key]) this.trails[cam][key] = { points: [], colorId };
    const entry = this.trails[cam][key];
    entry.colorId = colorId;
    const arr = entry.points;
    arr.push({ x: Number.NaN, y: Number.NaN });
    if (arr.length > MAX_POINTS) arr.shift();
  }

  migrate(cam: CameraKey, fromKey: string, toKey: string, toColorId: number): void {
    if (fromKey === toKey) {
      if (this.trails[cam][toKey]) this.trails[cam][toKey].colorId = toColorId;
      return;
    }
    const fromEntry = this.trails[cam][fromKey];
    const toEntry = this.trails[cam][toKey];
    if (fromEntry && !toEntry) {
      this.trails[cam][toKey] = { points: fromEntry.points, colorId: toColorId };
      delete this.trails[cam][fromKey];
      return;
    }
    if (fromEntry && toEntry) {
      toEntry.points.push({ x: Number.NaN, y: Number.NaN });
      toEntry.points.push(...fromEntry.points);
      toEntry.colorId = toColorId;
      while (toEntry.points.length > MAX_POINTS) toEntry.points.shift();
      delete this.trails[cam][fromKey];
    }
  }

  clearAll() {
    this.trails = { 'living-room': {}, 'kitchen': {}, 'family-room': {} };
  }
}

export function drawTrails(
  canvas: HTMLCanvasElement,
  cam: CameraKey,
  store: TrailStore,
  color: (id: number) => string,
  viewport?: { xMin: number; xMax: number; yMin: number; yMax: number; invertY?: boolean; drawCameraMarker?: boolean }
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const tracks = store.trails[cam];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  if (viewport) {
    minX = viewport.xMin; maxX = viewport.xMax;
    minY = viewport.yMin; maxY = viewport.yMax;
  } else {
    for (const key in tracks) {
      const pts = tracks[key]?.points || [];
      for (const p of pts) {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
      }
    }
    if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
      return;
    }
  }

  const pad = 10;
  const sx = (canvas.width - 2 * pad) / Math.max(1, maxX - minX);
  const sy = (canvas.height - 2 * pad) / Math.max(1, maxY - minY);

  ctx.strokeStyle = '#233142';
  ctx.lineWidth = 1;
  for (let gx = 0; gx < canvas.width; gx += 20) {
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, canvas.height); ctx.stroke();
  }
  for (let gy = 0; gy < canvas.height; gy += 20) {
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(canvas.width, gy); ctx.stroke();
  }

  for (const key in tracks) {
    const track = tracks[key];
    const pts = track?.points;
    if (!pts || pts.length === 0) continue;
    let open = false;
    let lastValid: Point | null = null;
    ctx.strokeStyle = color(track.colorId);
    ctx.lineWidth = 2;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const isGap = !Number.isFinite(p.x) || !Number.isFinite(p.y);
      if (isGap) {
        if (open) { ctx.stroke(); open = false; }
        continue;
      }
      const px = pad + (p.x - minX) * sx;
      const baseY = pad + (p.y - minY) * sy;
      const py = viewport?.invertY ? (canvas.height - baseY) : baseY;
      if (!open) { ctx.beginPath(); ctx.moveTo(px, py); open = true; }
      else { ctx.lineTo(px, py); }
      lastValid = p;
    }
    if (open) ctx.stroke();
    if (lastValid) {
      const hx = pad + (lastValid.x - minX) * sx;
      const baseHy = pad + (lastValid.y - minY) * sy;
      const hy = viewport?.invertY ? (canvas.height - baseHy) : baseHy;
      ctx.fillStyle = color(track.colorId);
      ctx.beginPath(); ctx.arc(hx, hy, 3, 0, Math.PI * 2); ctx.fill();
    }
  }

  // Optional camera marker at bottom center when viewport provided
  if (viewport?.drawCameraMarker) {
    const cx = canvas.width / 2;
    const cy = canvas.height - pad;
    ctx.fillStyle = '#ffcc00';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx - 8, cy - 12);
    ctx.lineTo(cx + 8, cy - 12);
    ctx.closePath();
    ctx.fill();
  }
}
