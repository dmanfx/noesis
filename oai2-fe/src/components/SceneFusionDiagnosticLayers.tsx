import { useEffect, useMemo, useRef, useState } from 'react';

import type {
  FloorplanLayer,
  FloorplanResponse,
} from './DepthDrawer';
import {
  bwColor,
  decodeFloat32,
  grayscaleColor,
  infernoColor,
  renderLayerToCanvas,
  renderStructuralFloorplanToCanvas,
  turboColor,
  viridisColor,
} from '../lib/renderUtils';
import { computeMaskedRange, type SimpleRange } from '../lib/depthQuality.js';
import { computeObservedFloorplanViewport } from '../lib/floorplanViewport.js';

type Props = {
  floorplan?: FloorplanResponse;
  enabled: boolean;
  drawerWidth: number;
};

const UNKNOWN_CELL_COLOR: [number, number, number, number] = [16, 22, 32, 255];
const UNKNOWN_CELL_ALT_COLOR: [number, number, number, number] = [20, 28, 39, 255];
const MASK_THRESHOLD = 1e-6;
const FURNITURE_RANGE: SimpleRange = { min: 0, max: 1.2 };
const ROOM_RANGE: SimpleRange = { min: 0, max: 2.5 };

function percentileSorted(sorted: number[], percentile: number): number {
  if (!sorted.length) return 0;
  if (sorted.length === 1) return sorted[0];
  const index = (Math.min(100, Math.max(0, percentile)) / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const fraction = index - lower;
  return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction);
}

function layerRange(layer?: FloorplanLayer): SimpleRange | null {
  const min = Number(layer?.value_min);
  const max = Number(layer?.value_max);
  return Number.isFinite(min) && Number.isFinite(max) && max > min
    ? { min, max }
    : null;
}

function layerAspect(layer?: FloorplanLayer): number {
  const rows = Number(layer?.grid_shape?.[0]);
  const columns = Number(layer?.grid_shape?.[1]);
  return Number.isFinite(rows) && Number.isFinite(columns) && rows > 0 && columns > 0
    ? columns / rows
    : 16 / 9;
}

function completeBounds(bounds?: FloorplanResponse['bounds']) {
  const minX = Number(bounds?.min_x);
  const maxX = Number(bounds?.max_x);
  const minZ = Number(bounds?.min_z);
  const maxZ = Number(bounds?.max_z);
  if (
    !Number.isFinite(minX)
    || !Number.isFinite(maxX)
    || !Number.isFinite(minZ)
    || !Number.isFinite(maxZ)
    || maxX <= minX
    || maxZ <= minZ
  ) {
    return undefined;
  }
  return { min_x: minX, max_x: maxX, min_z: minZ, max_z: maxZ };
}

function formatNumber(value?: number, digits = 2): string {
  if (value === undefined || !Number.isFinite(value)) return 'n/a';
  return value.toFixed(digits).replace(/\.00$/, '');
}

function generateGradient(palette: (t: number) => [number, number, number]): string {
  const stops: string[] = [];
  for (let index = 0; index <= 12; index += 1) {
    const t = index / 12;
    const [red, green, blue] = palette(t);
    stops.push(`rgb(${red}, ${green}, ${blue}) ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(to top, ${stops.join(', ')})`;
}

const turboGradient = generateGradient(turboColor);
const viridisGradient = generateGradient(viridisColor);
const infernoGradient = generateGradient(infernoColor);
const densityGradient = 'linear-gradient(to top, rgb(0,0,0) 0%, rgb(255,255,255) 100%)';

function Scale({
  gradient,
  range,
  unit = '',
}: {
  gradient: string;
  range: SimpleRange | null;
  unit?: string;
}) {
  const midpoint = range ? (range.min + range.max) / 2 : undefined;
  return (
    <div className="color-scale">
      <div className="color-scale__bar" style={{ background: gradient }} />
      <div className="color-scale__labels">
        <span className="color-scale__label color-scale__label--max">
          {range ? `${formatNumber(range.max)}${unit}` : 'n/a'}
        </span>
        <span className="color-scale__label">
          {midpoint === undefined ? '' : `${formatNumber(midpoint)}${unit}`}
        </span>
        <span className="color-scale__label color-scale__label--min">
          {range ? `${formatNumber(range.min)}${unit}` : 'n/a'}
        </span>
      </div>
    </div>
  );
}

function matchingMaskedRange(
  layer: FloorplanLayer | undefined,
  observed: FloorplanLayer | undefined,
): SimpleRange | null {
  if (!layer?.grid_b64 || !layer.grid_shape) return null;
  const values = decodeFloat32(layer.grid_b64);
  if (!values) return null;
  let mask: Float32Array | null = null;
  if (
    observed?.grid_b64
    && observed.grid_shape?.[0] === layer.grid_shape[0]
    && observed.grid_shape?.[1] === layer.grid_shape[1]
  ) {
    mask = decodeFloat32(observed.grid_b64);
  }
  return computeMaskedRange(values, {
    mask,
    maskThreshold: MASK_THRESHOLD,
    lowPercentile: 2,
    highPercentile: 98,
  });
}

function contrastRange(
  layer: FloorplanLayer | undefined,
  observed: FloorplanLayer | undefined,
): SimpleRange | null {
  if (!layer?.grid_b64 || !layer.grid_shape) return null;
  const values = decodeFloat32(layer.grid_b64);
  if (!values) return null;
  const mask = (
    observed?.grid_b64
    && observed.grid_shape?.[0] === layer.grid_shape[0]
    && observed.grid_shape?.[1] === layer.grid_shape[1]
  )
    ? decodeFloat32(observed.grid_b64)
    : null;
  const samples: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) continue;
    if (mask && (!Number.isFinite(mask[index]) || mask[index] <= MASK_THRESHOLD)) continue;
    samples.push(values[index]);
  }
  if (samples.length < 16) return null;
  samples.sort((left, right) => left - right);
  const min = percentileSorted(samples, 3);
  const max = percentileSorted(samples, 97);
  return max > min ? { min, max } : null;
}

export default function SceneFusionDiagnosticLayers({ floorplan, enabled, drawerWidth }: Props) {
  const [open, setOpen] = useState(false);
  const [aglRangeMode, setAglRangeMode] = useState<'furniture' | 'room' | 'auto'>('furniture');
  const structuralRef = useRef<HTMLCanvasElement | null>(null);
  const densityRef = useRef<HTMLCanvasElement | null>(null);
  const heightRef = useRef<HTMLCanvasElement | null>(null);
  const heightContrastRef = useRef<HTMLCanvasElement | null>(null);
  const heightAglRef = useRef<HTMLCanvasElement | null>(null);
  const distanceRef = useRef<HTMLCanvasElement | null>(null);
  const obstacleRef = useRef<HTMLCanvasElement | null>(null);
  const walkableRef = useRef<HTMLCanvasElement | null>(null);
  const gradientRef = useRef<HTMLCanvasElement | null>(null);

  const density = floorplan?.scene_fusion_diagnostic_density;
  const height = floorplan?.scene_fusion_diagnostic_height;
  const heightAgl = floorplan?.scene_fusion_diagnostic_height_agl;
  const distance = floorplan?.scene_fusion_diagnostic_distance;
  const gradient = floorplan?.scene_fusion_diagnostic_gradient;
  const obstacle = floorplan?.scene_fusion_diagnostic_obstacle_height;
  const walkable = floorplan?.scene_fusion_diagnostic_walkable;
  const structuralHeight = floorplan?.scene_fusion_diagnostic_structural_height;
  const surfaceObserved = floorplan?.scene_fusion_diagnostic_surface_observed;
  const roomFootprint = floorplan?.scene_fusion_diagnostic_room_footprint;
  const wallSupport = floorplan?.scene_fusion_diagnostic_wall_support;
  const roomBoundary = floorplan?.scene_fusion_diagnostic_room_boundary;
  const surfaceRgb = floorplan?.scene_fusion_diagnostic_surface_rgb;
  const meta = floorplan?.scene_fusion_meta;
  const diagnosticBounds = completeBounds(
    meta?.diagnostic_layers?.bounds ?? floorplan?.bounds,
  );
  const observed = surfaceObserved;

  const viewport = useMemo(() => {
    if (!observed?.grid_b64 || !observed.grid_shape) return null;
    const values = decodeFloat32(observed.grid_b64);
    if (!values) return null;
    return computeObservedFloorplanViewport({
      maskValues: values,
      rows: observed.grid_shape[0],
      cols: observed.grid_shape[1],
      maskInvert: false,
      maskThreshold: MASK_THRESHOLD,
      bounds: diagnosticBounds,
      paddingM: 0.3,
    });
  }, [diagnosticBounds, observed?.grid_b64, observed?.grid_shape]);
  const sourceRect = viewport?.sourceRect;
  const displayAspect = (layer?: FloorplanLayer) => (
    sourceRect && layer?.grid_shape?.[0] === observed?.grid_shape?.[0]
      && layer?.grid_shape?.[1] === observed?.grid_shape?.[1]
      ? sourceRect.width / sourceRect.height
      : layerAspect(layer)
  );

  const heightRange = layerRange(height);
  const heightContrastRange = useMemo(
    () => contrastRange(height, observed),
    [height?.grid_b64, height?.grid_shape, observed?.grid_b64, observed?.grid_shape],
  );
  const heightAglAutoRange = useMemo(
    () => matchingMaskedRange(heightAgl, observed),
    [heightAgl?.grid_b64, heightAgl?.grid_shape, observed?.grid_b64, observed?.grid_shape],
  );
  const heightAglRange = aglRangeMode === 'room'
    ? ROOM_RANGE
    : aglRangeMode === 'auto' && heightAglAutoRange
      ? heightAglAutoRange
      : FURNITURE_RANGE;

  useEffect(() => {
    if (!enabled || !open) return;
    const options = {
      fit: 'contain' as const,
      maskLayer: observed,
      maskThreshold: MASK_THRESHOLD,
      maskInvert: false,
      unknownColor: UNKNOWN_CELL_COLOR,
      unknownAltColor: UNKNOWN_CELL_ALT_COLOR,
      sourceRect,
      imageSmoothing: false,
    };
    renderLayerToCanvas(densityRef.current, density, grayscaleColor, options);
    renderLayerToCanvas(heightRef.current, height, infernoColor, options);
    renderLayerToCanvas(heightContrastRef.current, height, turboColor, {
      ...options,
      valueMin: heightContrastRange?.min,
      valueMax: heightContrastRange?.max,
      gamma: 0.9,
    });
    renderLayerToCanvas(heightAglRef.current, heightAgl, turboColor, {
      ...options,
      valueMin: heightAglRange.min,
      valueMax: heightAglRange.max,
    });
    renderLayerToCanvas(distanceRef.current, distance, viridisColor, options);
    renderLayerToCanvas(obstacleRef.current, obstacle, infernoColor, options);
    renderLayerToCanvas(walkableRef.current, walkable, bwColor, options);
    renderLayerToCanvas(gradientRef.current, gradient, viridisColor, options);
    renderStructuralFloorplanToCanvas(
      structuralRef.current,
      structuralHeight,
      roomFootprint,
      surfaceObserved,
      wallSupport,
      roomBoundary,
      surfaceRgb,
      {
        fit: 'contain',
        sourceRect,
        bounds: diagnosticBounds,
        metricGridM: 1,
        imageSmoothing: false,
      },
    );
  }, [
    aglRangeMode,
    density,
    distance,
    drawerWidth,
    enabled,
    diagnosticBounds,
    gradient,
    height,
    heightAgl,
    heightAglRange.max,
    heightAglRange.min,
    heightContrastRange?.max,
    heightContrastRange?.min,
    obstacle,
    observed,
    open,
    roomBoundary,
    roomFootprint,
    sourceRect,
    structuralHeight,
    surfaceObserved,
    surfaceRgb,
    walkable,
    wallSupport,
  ]);

  if (!enabled) return null;

  const diagnosticMeta = meta?.diagnostic_layers;
  const viewCount = (diagnosticMeta?.fixed_anchor_view_count ?? 0)
    + (diagnosticMeta?.phone_view_count ?? 0);

  return (
    <details
      className="depth-diagnostics scene-fusion-diagnostics"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>Joint static + phone diagnostic layers</summary>
      <div className="scene-fusion-diagnostics__provenance">
        {viewCount > 0 ? `${viewCount}-view MA cohort` : 'Joint MA cohort'} · cached admitted common-frame result · opening this view runs no inference or registration
      </div>
      <div className="depth-diagnostics__toolbar">
        <label>
          <span>AGL range</span>
          <select
            value={aglRangeMode}
            onChange={(event) => setAglRangeMode(event.target.value as 'furniture' | 'room' | 'auto')}
          >
            <option value="furniture">Furniture (0–1.2 m)</option>
            <option value="room">Room (0–2.5 m)</option>
            <option value="auto">Auto (p2–p98)</option>
          </select>
        </label>
      </div>
      <div className="heatmap-grid">
        <div className="heatmap-cell heatmap-cell--primary heatmap-cell--no-scale">
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Structural Composite (Joint Diagnostic)</div>
            <canvas ref={structuralRef} className="heatmap-canvas heatmap-canvas--semantic" style={{ aspectRatio: `${displayAspect(structuralHeight)}` }} />
            <div className="unknown-cell-key">
              Dominant sub-ceiling surfaces, accepted RGB voxels, measured support, and observed boundary
            </div>
          </div>
        </div>
        <div className="heatmap-cell heatmap-cell--left">
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Density (Grayscale)</div>
            <canvas ref={densityRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(density)}` }} />
          </div>
          <div className="heatmap-cell__scale"><Scale gradient={densityGradient} range={{ min: 0, max: 1 }} /></div>
        </div>
        <div className="heatmap-cell heatmap-cell--right">
          <div className="heatmap-cell__scale"><Scale gradient={infernoGradient} range={heightRange} unit=" m" /></div>
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Surface Height (Inferno)</div>
            <canvas ref={heightRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(height)}` }} />
          </div>
        </div>
        <div className="heatmap-cell heatmap-cell--right">
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Surface Height (Contrast)</div>
            <canvas ref={heightContrastRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(height)}` }} />
          </div>
          <div className="heatmap-cell__scale"><Scale gradient={turboGradient} range={heightContrastRange} unit=" m" /></div>
        </div>
        <div className="heatmap-cell heatmap-cell--left">
          <div className="heatmap-cell__scale"><Scale gradient={turboGradient} range={heightAglRange} unit=" m" /></div>
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Height Above Floor</div>
            <canvas ref={heightAglRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(heightAgl)}` }} />
          </div>
        </div>
        <div className="heatmap-cell heatmap-cell--right">
          <div className="heatmap-cell__scale"><Scale gradient={viridisGradient} range={layerRange(distance)} unit=" m" /></div>
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Camera Range (Viridis)</div>
            <canvas ref={distanceRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(distance)}` }} />
          </div>
        </div>
        <div className="heatmap-cell heatmap-cell--left">
          <div className="heatmap-cell__scale"><Scale gradient={infernoGradient} range={layerRange(obstacle)} unit=" m" /></div>
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Obstacle Height (Clean)</div>
            <canvas ref={obstacleRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(obstacle)}` }} />
          </div>
        </div>
        <div className="heatmap-cell heatmap-cell--right">
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Measured Walkable Support (Binary)</div>
            <canvas ref={walkableRef} className="heatmap-canvas heatmap-canvas--semantic" style={{ aspectRatio: `${displayAspect(walkable)}` }} />
          </div>
          <div className="heatmap-cell__scale"><Scale gradient={densityGradient} range={{ min: 0, max: 1 }} /></div>
        </div>
        <div className="heatmap-cell heatmap-cell--right">
          <div className="heatmap-cell__body">
            <div className="heatmap-cell__title">Gradient (Edges)</div>
            <canvas ref={gradientRef} className="heatmap-canvas" style={{ aspectRatio: `${displayAspect(gradient)}` }} />
          </div>
          <div className="heatmap-cell__scale"><Scale gradient={viridisGradient} range={{ min: 0, max: 1 }} /></div>
        </div>
      </div>
      <div className="primitives-submeta">
        Fixed anchor geometry stays calibrated; joint-pass phone geometry reuses the admitted bounded refinement. These layers are review-only and do not update Heatmap, BEV, tracking, or navigation state.
      </div>
    </details>
  );
}
