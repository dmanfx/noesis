import React, { useEffect, useMemo, useRef } from 'react';
import { decodeFloat32 } from '../lib/rasterEncoding';
import { normalizeBevTrailConfig, type BevTrailConfig } from '../lib/bevTrails';
import { colorForTrack } from '../lib/camera';
import {
  fitUnifiedBevViewport, unifiedBevLabelPlacement, unifiedBevRasterTransform, unifiedBevStatus, unifiedBevTrailSegments,
  type UnifiedBevDisplayState,
} from '../lib/unifiedBevConsumer.js';

type Props = { display: UnifiedBevDisplayState; connection: string; trailEnabled: boolean;
  trailConfig?: Partial<BevTrailConfig>; debug?: boolean };

const identityColor = (entityId: string, stableId: number | null) => {
  let id = stableId ?? 0;
  if (stableId === null) for (const c of entityId) id = (id * 31 + c.charCodeAt(0)) >>> 0;
  return colorForTrack(id);
};

export function UnifiedBevView({ display, connection, trailEnabled, trailConfig, debug = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cfg = useMemo(() => normalizeBevTrailConfig({ ...trailConfig, enabled: trailEnabled }), [trailConfig, trailEnabled]);
  const { world } = display;
  // Decode each accepted static raster once, outside the changing person draw.
  const staticImages = useMemo(() => display.rasters.map(raster => {
    const values = decodeFloat32(raster.layer.grid_b64);
    const mask = decodeFloat32(raster.mask.grid_b64);
    const [rows, cols] = raster.layer.grid_shape;
    if (!values || !mask || values.length !== rows * cols || mask.length !== values.length) return { raster, image: null };
    const image = document.createElement('canvas'); image.width = cols; image.height = rows;
    const ctx = image.getContext('2d');
    if (!ctx) return { raster, image: null };
    const pixels = ctx.createImageData(cols, rows);
    for (let i = 0; i < values.length; i++) {
      if (!Number.isFinite(values[i]) || !Number.isFinite(mask[i]) || mask[i] <= 0) continue;
      const height = Math.min(1, Math.max(0, values[i] / 2.5));
      pixels.data.set([Math.round(116 + height * 128), Math.round(134 + height * 114),
        Math.round(154 + height * 100), 255], i * 4);
    }
    ctx.putImageData(pixels, 0, 0);
    return { raster, image };
  }), [display.rasters]);
  const imageUnavailable = staticImages.some(v => !v.image) || world.maps.some(m => (
    !staticImages.some(v => v.raster.cameraId === m.cameraId && v.raster.mapId === m.mapId)
  ));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2, 4096 / Math.max(width, height));
      canvas.width = Math.max(1, Math.round(width * dpr)); canvas.height = Math.max(1, Math.round(height * dpr));
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      const viewport = fitUnifiedBevViewport(world, width, height);
      if (!viewport) return;
      for (const map of world.maps) {
        const cached = staticImages.find(v => v.raster.cameraId === map.cameraId
          && v.raster.mapId === map.mapId && v.raster.snapshotContentSha256 === map.snapshotContentSha256);
        if (cached?.image) {
          ctx.save(); ctx.transform(...unifiedBevRasterTransform(map, viewport) as [number, number, number, number, number, number]);
          ctx.imageSmoothingEnabled = true; ctx.drawImage(cached.image, 0, 0); ctx.restore();
        }
        if (debug) {
          ctx.beginPath();
          map.sceneCorners.forEach((p, i) => { const [x, y] = viewport.point(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.closePath(); ctx.strokeStyle = 'rgba(132,158,180,.5)'; ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      for (const person of world.people) {
        const color = identityColor(person.entityId, person.stableId);
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        const segments = unifiedBevTrailSegments(person, world, cfg);
        for (const segment of segments) {
          if (segment.length < 2) continue;
          ctx.globalAlpha = Math.max(cfg.min_alpha, .65); ctx.beginPath();
          segment.forEach((p, i) => { const [x, y] = viewport.point(p.position); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        const [x, y] = viewport.point(person.position);
        ctx.setLineDash(person.positionState === 'predicted' ? [3, 2] : []);
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fillStyle = person.positionState === 'held' ? '#0b1118' : color;
        ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
        const label = `${person.label}${person.positionState !== 'measured' ? ` · ${person.positionState}` : ''}`;
        ctx.font = '12px system-ui, sans-serif';
        const placement = unifiedBevLabelPlacement([x, y], width, height, ctx.measureText(label).width);
        ctx.textAlign = placement.textAlign;
        ctx.fillStyle = 'rgba(8,15,24,.9)';
        ctx.fillRect(placement.boxLeft, placement.boxTop, placement.boxWidth, placement.boxHeight);
        ctx.fillStyle = '#e4eef6'; ctx.fillText(label, placement.textX, placement.textY);
      }
    };
    draw();
    const resize = new ResizeObserver(draw); resize.observe(canvas);
    return () => resize.disconnect();
  }, [world, staticImages, cfg, debug]);

  const status = unifiedBevStatus(display, connection);
  const counts = ['measured', 'predicted', 'held'].map(s => ({ state: s, count: world.people.filter(p => p.positionState === s).length }));
  return (
    <section className="panel unified-bev-card" aria-labelledby="unified-bev-title"
      data-authored-scene-sha256={world.authoredSceneSha256 ?? ''} data-snapshot-id={world.cursor?.snapshotId ?? ''}
      data-world-sequence={world.cursor?.sequence ?? ''} data-person-count={world.people.length}
      data-map-count={world.maps.length} data-status={status}>
      <div className="unified-bev-head">
        <h2 id="unified-bev-title">Unified BEV</h2>
        <span className="unified-bev-status">{status}</span>
      </div>
      <div className="unified-bev-canvas-wrap">
        <canvas ref={canvasRef} aria-label={`Unified map. ${status}. ${world.maps.length} admitted maps.`} />
        {!world.maps.length && !world.people.length && <div className="unified-bev-placeholder">
          {connection !== 'open' ? 'Waiting for connection' : !world.bindings ? 'Scene mapping unavailable' : 'Waiting for map'}
        </div>}
      </div>
      <div className="unified-bev-caption">
        <span>{world.maps.length ? `${world.maps.length} static maps` : 'Map unavailable'}{imageUnavailable ? ' · map image unavailable' : ''}</span>
        <span>{counts.filter(c => c.count).map(c => `${c.count} ${c.state}`).join(' · ') || (display.hasSnapshot ? 'No current positions' : 'Waiting for canonical snapshot')}</span>
      </div>
      <ul className="unified-bev-accessible-people" aria-label="Canonical people">
        {world.people.map(person => <li key={person.entityId} data-entity-id={person.entityId}
          data-scene-position={JSON.stringify(person.position)} data-world-position={JSON.stringify(person.worldPosition)}
          data-position-state={person.positionState} data-observed-at-us={person.observedAtUs}
          data-world-revision={person.worldFrameRevision} data-transform-sha256={person.worldTransformSha256 ?? ''}
          data-scene-mapping-sha256={person.sceneMappingSha256} data-scene-transform-count={person.sceneTransformCount}
          data-cohort={JSON.stringify(person.cohort)} data-trail-count={person.trail.length}>
          {person.label}: {person.positionState}
        </li>)}
      </ul>
      {debug && <div className="unified-bev-diagnostic">{world.diagnostic.reason}{world.unavailable.map(p => ` · ${p.entityId}: ${p.reason}`).join('')}</div>}
    </section>
  );
}
