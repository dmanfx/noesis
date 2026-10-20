import type { UnifiedBevState, UnifiedBevMap, UnifiedBevPerson, UnifiedBevTrailPoint, Vec3 } from './unifiedBevWorld.js';
import type { BevTrailConfig } from './bevTrails';
export const UNIFIED_BEV_AUTHORED_SCENE_SHA256: string;
export type UnifiedBevRasterLayer = Readonly<{ grid_b64: string; grid_shape: readonly number[]; grid_encoding?: string }>;
export type UnifiedBevRgbLayer = Readonly<{ rgb_b64: string; rgb_shape: readonly number[]; rgb_encoding?: string;
  observed_b64: string; observed_encoding?: string }>;
export type UnifiedBevRaster = Readonly<{ cameraId: string; mapId: string; snapshotContentSha256: string;
  layer: UnifiedBevRasterLayer | null; heightLayer: UnifiedBevRasterLayer | null;
  mask: UnifiedBevRasterLayer | null; surfaceRgb: UnifiedBevRgbLayer | null }>;
export type UnifiedBevDisplayState = Readonly<{
  world: UnifiedBevState; hasSnapshot: boolean;
  rasters: readonly UnifiedBevRaster[];
}>;
export type UnifiedBevViewport = Readonly<{ minX: number; maxX: number; minZ: number; maxZ: number; scale: number;
  point: (position: Vec3) => [number, number] }>;
export function dispatchUnifiedBevMessage(message: unknown,
  receive: ((message: unknown, receivedAtMs: number) => void) | undefined, receivedAtMs: number): boolean;
export function createUnifiedBevDisplayState(): UnifiedBevDisplayState;
export function createUnifiedBevConsumer(options?: {
  onChange?: (display: UnifiedBevDisplayState) => void; now?: () => number;
  schedule?: (fn: () => void, delay: number) => unknown; cancel?: (timer: unknown) => void;
}): Readonly<{ getSnapshot: () => UnifiedBevDisplayState;
  receive: (message: unknown, receivedAtMs?: number) => UnifiedBevDisplayState;
  disconnect: (reason?: string) => void; stop: () => void }>;
export function fitUnifiedBevViewport(world: UnifiedBevState, width: number, height: number, padding?: number): UnifiedBevViewport | null;
export function unifiedBevLabelPlacement(marker: readonly [number, number], width: number, height: number,
  textWidth: number): Readonly<{ boxLeft: number; boxTop: number; boxWidth: number; boxHeight: number;
    textX: number; textY: number; textAlign: 'right' }>;
export function unifiedBevRasterTransform(map: UnifiedBevMap, viewport: UnifiedBevViewport): number[];
export function unifiedBevTrailSegments(person: UnifiedBevPerson, world: UnifiedBevState, config: BevTrailConfig): UnifiedBevTrailPoint[][];
export function unifiedBevStatus(display: UnifiedBevDisplayState, connection: string): string;
