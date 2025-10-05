import { CameraKey } from './camera';

type Point = { x: number; y: number };
type Trails = Record<CameraKey, Record<number, Point[]>>;

const MAX_POINTS = 300;

export class TrailStore {
  trails: Trails = { 'living-room': {}, 'kitchen': {}, 'family-room': {} };

  push(cam: CameraKey, trackId: number, pt: Point) {
    if (!this.trails[cam][trackId]) this.trails[cam][trackId] = [];
    const arr = this.trails[cam][trackId];
    arr.push(pt);
    if (arr.length > MAX_POINTS) arr.shift();
  }

  // Insert a gap marker so drawTrails breaks the polyline for this track
  pushBreak(cam: CameraKey, trackId: number) {
    if (!this.trails[cam][trackId]) this.trails[cam][trackId] = [];
    const arr = this.trails[cam][trackId];
    arr.push({ x: Number.NaN, y: Number.NaN });
    if (arr.length > MAX_POINTS) arr.shift();
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
    for (const tidStr in tracks) {
      const pts = tracks[Number(tidStr)] || [];
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

  // Lightweight, throttled instrumentation to validate draw pipeline
  const now = Date.now();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const lastLogMap: any = (drawTrails as any)._lastLogMap || ((drawTrails as any)._lastLogMap = {});
  const lastLogTs: number = lastLogMap[cam] || 0;
  let trackCount = 0;
  let totalPoints = 0;
  let validPoints = 0;
  let outXLo = 0, outXHi = 0, outYLo = 0, outYHi = 0;

  for (const tidStr in tracks) {
    const tid = Number(tidStr);
    const pts = tracks[tid];
    if (!pts || pts.length === 0) continue;
    trackCount++;
    let open = false;
    let lastValid: Point | null = null;
    ctx.strokeStyle = color(tid);
    ctx.lineWidth = 2;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      totalPoints++;
      const isGap = !Number.isFinite(p.x) || !Number.isFinite(p.y);
      if (isGap) {
        if (open) { ctx.stroke(); open = false; }
        continue;
      }
      validPoints++;
      // Count points outside viewport extents (not clamped)
      if (p.x < minX) outXLo++; else if (p.x > maxX) outXHi++;
      if (p.y < minY) outYLo++; else if (p.y > maxY) outYHi++;
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
      ctx.fillStyle = color(tid);
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

  // Emit a concise log at most every 2s per camera
  if (viewport && now - lastLogTs > 2000) {
    // eslint-disable-next-line no-console
    console.info(
      `[trails] ${cam}: tracks=${trackCount}, points=${validPoints}/${totalPoints}, ` +
      `extents=([${minX.toFixed(2)},${minY.toFixed(2)}]-[${maxX.toFixed(2)},${maxY.toFixed(2)}]), ` +
      `canvas=${canvas.width}x${canvas.height}, outX:<${outXLo} >${outXHi}, outY:<${outYLo} >${outYHi}`
    );
    lastLogMap[cam] = now;
  }
}
