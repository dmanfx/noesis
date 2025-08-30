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

  clearAll() {
    this.trails = { 'living-room': {}, 'kitchen': {}, 'family-room': {} };
  }
}

export function drawTrails(
  canvas: HTMLCanvasElement,
  cam: CameraKey,
  store: TrailStore,
  color: (id: number) => string
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const tracks = store.trails[cam];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
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

  for (const tidStr in tracks) {
    const tid = Number(tidStr);
    const pts = tracks[tid];
    if (!pts || pts.length < 2) continue;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const px = pad + (pts[i].x - minX) * sx;
      const py = pad + (pts[i].y - minY) * sy;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color(tid);
    ctx.lineWidth = 2;
    ctx.stroke();
    const last = pts[pts.length - 1];
    const hx = pad + (last.x - minX) * sx;
    const hy = pad + (last.y - minY) * sy;
    ctx.fillStyle = color(tid);
    ctx.beginPath(); ctx.arc(hx, hy, 3, 0, Math.PI * 2); ctx.fill();
  }
}

