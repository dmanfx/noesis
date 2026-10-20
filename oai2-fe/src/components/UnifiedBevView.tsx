import React, { useEffect, useMemo, useRef } from 'react';
import { normalizeBevTrailConfig, type BevTrailConfig } from '../lib/bevTrails';
import { colorForTrack } from '../lib/camera';
import type { PresentationSettings } from '../lib/dashboardPresentation.js';
import {
  BEV_LAYER_LABELS, BEV_PRESENTATION_IMAGE_LIMIT, bevPaletteColor,
  combinedBevHeightRange, createBevPresentationRaster,
} from '../lib/bevPresentationRaster';
import {
  fitUnifiedBevViewport, unifiedBevLabelPlacement, unifiedBevRasterTransform, unifiedBevStatus, unifiedBevTrailSegments,
  type UnifiedBevDisplayState,
} from '../lib/unifiedBevConsumer.js';

type Props = { display: UnifiedBevDisplayState; connection: string; trailEnabled: boolean;
  trailConfig?: Partial<BevTrailConfig>; debug?: boolean; children?: React.ReactNode;
  presentation?: Partial<PresentationSettings> };

const identityColor = (entityId: string, stableId: number | null) => {
  let id = stableId ?? 0;
  if (stableId === null) for (const c of entityId) id = (id * 31 + c.charCodeAt(0)) >>> 0;
  return colorForTrack(id);
};

export function UnifiedBevView({ display, connection, trailEnabled, trailConfig, debug = false, children, presentation }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const layer = presentation?.bevLayer ?? 'structure';
  const palette = presentation?.palette ?? 'turbo';
  const opacity = Number.isFinite(presentation?.opacity) ? Math.min(1, Math.max(0, presentation!.opacity!)) : 1;
  const showPeople = presentation?.people !== false;
  const showLabels = presentation?.labels !== false;
  // Canonical history already reaches this consumer. The local display choice
  // is independent of the legacy runtime/video trail switch.
  const showTrails = showPeople && (presentation?.trails ?? trailEnabled);
  const cfg = useMemo(() => normalizeBevTrailConfig({ ...trailConfig, enabled: showTrails }), [trailConfig, showTrails]);
  const { world } = display;
  // These caches have only currently admitted maps. Tracking and appearance
  // changes reuse decoded data; at most seven rendered images live per map.
  const staticRasters = useMemo(() => display.rasters.flatMap(raster => {
    const map = world.maps.find(m => m.cameraId === raster.cameraId && m.mapId === raster.mapId
      && m.snapshotContentSha256 === raster.snapshotContentSha256);
    return map ? [{ raster, decoded: createBevPresentationRaster(raster, map), images: new Map<string, HTMLCanvasElement | null>() }] : [];
  }), [display.rasters]);
  const heightRange = useMemo(() => combinedBevHeightRange(staticRasters.map(entry => entry.decoded.heightRange)), [staticRasters]);
  const staticImages = useMemo(() => staticRasters.map(entry => {
    const key = layer === 'height' ? `${layer}:${palette}:${heightRange?.min}:${heightRange?.max}`
      : layer === 'structure' ? `${layer}:${presentation?.theme === 'light' ? 'light' : 'dark'}` : layer;
    if (!entry.images.has(key)) {
      const rendered = entry.decoded.render(layer, palette, heightRange, presentation?.theme);
      let image: HTMLCanvasElement | null = null;
      if (rendered) {
        image = document.createElement('canvas'); image.width = rendered.width; image.height = rendered.height;
        const context = image.getContext('2d');
        if (context) {
          const pixels = context.createImageData(rendered.width, rendered.height);
          pixels.data.set(rendered.pixels); context.putImageData(pixels, 0, 0);
        } else image = null;
      }
      if (entry.images.size >= BEV_PRESENTATION_IMAGE_LIMIT) entry.images.delete(entry.images.keys().next().value!);
      entry.images.set(key, image);
    }
    return { raster: entry.raster, image: entry.images.get(key) ?? null };
  }), [staticRasters, layer, palette, heightRange, presentation?.theme]);
  const unavailableMaps = world.maps.filter(map => !staticImages.some(entry => entry.image
    && entry.raster.cameraId === map.cameraId && entry.raster.mapId === map.mapId
    && entry.raster.snapshotContentSha256 === map.snapshotContentSha256));
  const heightGradient = useMemo(() => `linear-gradient(to right, ${Array.from({ length: 17 }, (_, i) => {
    const rgb = bevPaletteColor(palette, i / 16); return `rgb(${rgb.join(',')}) ${i / 16 * 100}%`;
  }).join(', ')})`, [palette]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (width <= 0 || height <= 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2, 4096 / Math.max(width, height));
      canvas.width = Math.max(1, Math.round(width * dpr)); canvas.height = Math.max(1, Math.round(height * dpr));
      const ctx = canvas.getContext('2d'); if (!ctx) return;
      const style = typeof window.getComputedStyle === 'function' ? window.getComputedStyle(canvas) : null;
      const token = (specific: string, general: string, fallback: string) => style?.getPropertyValue(specific).trim()
        || style?.getPropertyValue(general).trim() || fallback;
      const labelText = token('--bev-label-text', '--text', '#e4eef6');
      const labelBackground = token('--bev-label-background', '--panel', 'rgba(8,15,24,.9)');
      const heldFill = token('--bev-held-fill', '--panel', '#0b1118');
      const mapOutline = token('--bev-map-outline', '--border', 'rgba(132,158,180,.5)');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, width, height);
      const viewport = fitUnifiedBevViewport(world, width, height);
      if (!viewport) return;
      for (const map of world.maps) {
        const cached = staticImages.find(v => v.raster.cameraId === map.cameraId
          && v.raster.mapId === map.mapId && v.raster.snapshotContentSha256 === map.snapshotContentSha256);
        if (cached?.image) {
          ctx.save(); ctx.transform(...unifiedBevRasterTransform(map, viewport) as [number, number, number, number, number, number]);
          ctx.globalAlpha = opacity; ctx.imageSmoothingEnabled = layer === 'structure' || layer === 'texture';
          ctx.drawImage(cached.image, 0, 0); ctx.restore();
        }
        if (debug) {
          ctx.beginPath();
          map.sceneCorners.forEach((p, i) => { const [x, y] = viewport.point(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
          ctx.closePath(); ctx.strokeStyle = mapOutline; ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      ctx.globalAlpha = 1;
      for (const person of world.people) {
        if (!showPeople) continue;
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
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fillStyle = person.positionState === 'held' ? heldFill : color;
        ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
        if (!showLabels) continue;
        const label = `${person.label}${person.positionState !== 'measured' ? ` · ${person.positionState}` : ''}`;
        ctx.font = '12px system-ui, sans-serif';
        const placement = unifiedBevLabelPlacement([x, y], width, height, ctx.measureText(label).width);
        ctx.textAlign = placement.textAlign;
        ctx.fillStyle = labelBackground;
        ctx.fillRect(placement.boxLeft, placement.boxTop, placement.boxWidth, placement.boxHeight);
        ctx.fillStyle = labelText; ctx.fillText(label, placement.textX, placement.textY);
      }
    };
    draw();
    const resize = new ResizeObserver(draw); resize.observe(canvas);
    return () => resize.disconnect();
  }, [world, staticImages, cfg, debug, opacity, showPeople, showLabels, layer, presentation?.theme]);

  const status = unifiedBevStatus(display, connection);
  const counts = ['measured', 'predicted', 'held'].map(s => ({ state: s, count: world.people.filter(p => p.positionState === s).length }));
  return (
    <section className="panel unified-bev-card" aria-labelledby="unified-bev-title"
      data-authored-scene-sha256={world.authoredSceneSha256 ?? ''} data-snapshot-id={world.cursor?.snapshotId ?? ''}
      data-world-sequence={world.cursor?.sequence ?? ''} data-person-count={world.people.length}
      data-map-count={world.maps.length} data-status={status} data-bev-layer={layer} data-bev-palette={palette}
      data-show-people={showPeople} data-show-trails={showTrails} data-show-labels={showLabels} data-map-opacity={opacity}>
      <div className="unified-bev-head">
        <h2 id="unified-bev-title">Unified BEV</h2>
        <span className="unified-bev-status">{status}</span>
      </div>
      <div className="unified-bev-canvas-wrap">
        <canvas ref={canvasRef} aria-label={`Unified map. ${BEV_LAYER_LABELS[layer]}. ${status}. ${world.maps.length} admitted maps.`} />
        {!world.maps.length && !world.people.length && <div className="unified-bev-placeholder">
          {connection !== 'open' ? 'Waiting for connection' : !world.bindings ? 'Scene mapping unavailable' : 'Waiting for map'}
        </div>}
        {children}
      </div>
      <div className="unified-bev-legend" aria-label={`${BEV_LAYER_LABELS[layer]} legend`}>
        {layer === 'structure' && <span>Structure · observed PCF geometry</span>}
        {layer === 'texture' && <span>Texture · observed PCF surface RGB</span>}
        {layer === 'height' && <>
          <span>Height above ground (metres)</span>
          {heightRange && <><span className="unified-bev-legend-gradient" aria-hidden="true" style={{ background: heightGradient }} />
            <span>{heightRange.min.toFixed(2)} m – {heightRange.max.toFixed(2)} m · {palette}</span></>}
        </>}
        {layer === 'normals' && <>
          <span>Room normals · derived from PCF heightfield</span>
          <span>RGB orientation in scene axes: <i style={{ background: '#ff8080' }} aria-hidden="true" /> +X,
            <i style={{ background: '#80ff80' }} aria-hidden="true" /> +Y,
            <i style={{ background: '#8080ff' }} aria-hidden="true" /> +Z</span>
        </>}
      </div>
      {unavailableMaps.length > 0 && <ul className="unified-bev-layer-unavailable" aria-label="Unavailable map layers">
        {unavailableMaps.map(map => <li key={map.cameraId} data-camera-id={map.cameraId}>
          {map.cameraId}: {BEV_LAYER_LABELS[layer]} unavailable for this admitted map
        </li>)}
      </ul>}
      <div className="unified-bev-caption">
        <span>{world.maps.length ? `${world.maps.length} static maps` : 'Map unavailable'}{unavailableMaps.length ? ' · map image unavailable' : ''}</span>
        <span>{counts.filter(c => c.count).map(c => `${c.count} ${c.state}`).join(' · ') || (display.hasSnapshot ? 'No current positions' : 'Waiting for canonical snapshot')}</span>
      </div>
      {showPeople && <ul className="unified-bev-accessible-people" aria-label="Canonical people">
        {world.people.map(person => <li key={person.entityId} data-entity-id={person.entityId}
          data-scene-position={JSON.stringify(person.position)} data-world-position={JSON.stringify(person.worldPosition)}
          data-position-state={person.positionState} data-observed-at-us={person.observedAtUs}
          data-world-revision={person.worldFrameRevision} data-transform-sha256={person.worldTransformSha256 ?? ''}
          data-scene-mapping-sha256={person.sceneMappingSha256} data-scene-transform-count={person.sceneTransformCount}
          data-cohort={JSON.stringify(person.cohort)} data-trail-count={person.trail.length}>
          {person.label}: {person.positionState}
        </li>)}
      </ul>}
      {debug && <div className="unified-bev-diagnostic">{world.diagnostic.reason}{world.unavailable.map(p => ` · ${p.entityId}: ${p.reason}`).join('')}</div>}
    </section>
  );
}
