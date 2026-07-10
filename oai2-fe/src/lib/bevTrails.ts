export type TrailPoint = { x: number; y: number; t: number };

export type TrailTrack = {
  points: TrailPoint[];
  lastSeen: number;
  label: string;
  colorId: number;
  // Fast head smoothing for responsive dot motion.
  emaX?: number;
  emaY?: number;
  emaTs?: number;
  // Separate trail smoothing so line tails can stay steadier than the head dot.
  trailEmaX?: number;
  trailEmaY?: number;
  trailEmaTs?: number;
};

export type BevTrailConfig = {
  enabled: boolean;
  window_s: number;
  draw_stride: number;
  min_step_px: number;
  // Legacy shared smoothing knob (kept for backward compatibility with WS payloads).
  min_dt_s: number;
  smooth_tau_s: number;
  // BEV-specific smoothing split: responsive head + steadier trail tail.
  head_smooth_tau_s: number;
  trail_smooth_tau_s: number;
  // Clamp scene-space decimation to avoid starving trails on wide spans.
  min_step_scene_floor: number;
  min_step_scene_ceil: number;
  max_speed_px_per_s: number;
  max_points_per_track: number;
  max_tracks: number;
  min_alpha: number;
  gap_ms: number;
  stale_blink_start_ms: number;
  stale_blink_period_ms: number;
  stale_head_blink_enabled: boolean;
  reset_after_s: number;
  teleport_break_ratio: number;
};

export const DEFAULT_BEV_TRAIL_CONFIG: BevTrailConfig = {
  enabled: true,
  window_s: 8.0,
  draw_stride: 1,
  min_step_px: 0.2,
  min_dt_s: 0.03,
  smooth_tau_s: 0.14,
  head_smooth_tau_s: 0.06,
  trail_smooth_tau_s: 0.19,
  min_step_scene_floor: 0.0,
  min_step_scene_ceil: 0.11,
  max_speed_px_per_s: 1800.0,
  max_points_per_track: 257,
  max_tracks: 8,
  min_alpha: 0.12,
  gap_ms: 650,
  stale_blink_start_ms: 700,
  stale_blink_period_ms: 1400,
  stale_head_blink_enabled: false,
  reset_after_s: 1.25,
  teleport_break_ratio: 4.0,
};

type BoundsLike = {
  xMin: number;
  xMax: number;
  zMin: number;
  zMax: number;
  widthPx: number;
  heightPx: number;
};

type UpsertTrailSampleArgs = {
  nowMs: number;
  x: number;
  y: number;
  doSample: boolean;
  sceneUnitsPerPx: number;
  cfg: BevTrailConfig;
};

type UpsertTrailSampleResult = {
  headX: number;
  headY: number;
  sampled: boolean;
};

const clampNumber = (value: unknown, fallback: number): number => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const clampBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') return value;
  if (value == null) return fallback;
  const text = String(value).trim().toLowerCase();
  if (text === '1' || text === 'true' || text === 'yes' || text === 'on') return true;
  if (text === '0' || text === 'false' || text === 'no' || text === 'off') return false;
  return fallback;
};

const isFinitePoint = (point?: TrailPoint | null): point is TrailPoint => {
  if (!point) return false;
  return Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.t);
};

const findLastFinitePoint = (points: TrailPoint[]): { index: number; point: TrailPoint } | null => {
  for (let i = points.length - 1; i >= 0; i -= 1) {
    const point = points[i];
    if (isFinitePoint(point)) return { index: i, point };
  }
  return null;
};

const pushGap = (points: TrailPoint[], nowMs: number): void => {
  const last = points.length ? points[points.length - 1] : null;
  if (!last || isFinitePoint(last)) {
    points.push({ x: Number.NaN, y: Number.NaN, t: nowMs });
  }
};

const enforcePointCap = (points: TrailPoint[], cap: number): void => {
  while (points.length > cap) points.shift();
};

const pruneTrackWindow = (track: TrailTrack, nowMs: number, cfg: BevTrailConfig): void => {
  const windowMs = Math.max(100, cfg.window_s * 1000);
  while (track.points.length && (nowMs - track.points[0].t) > windowMs) {
    track.points.shift();
  }
  enforcePointCap(track.points, Math.max(2, cfg.max_points_per_track));
  while (track.points.length && !isFinitePoint(track.points[0])) {
    track.points.shift();
  }
};

export const normalizeBevTrailConfig = (raw?: Partial<BevTrailConfig>): BevTrailConfig => {
  const rawCfg = raw || {};
  const cfg = { ...DEFAULT_BEV_TRAIL_CONFIG, ...rawCfg };
  const legacyTau = Math.max(0.0, clampNumber(cfg.smooth_tau_s, DEFAULT_BEV_TRAIL_CONFIG.smooth_tau_s));
  const hasHeadTau = Object.prototype.hasOwnProperty.call(rawCfg, 'head_smooth_tau_s');
  const hasTrailTau = Object.prototype.hasOwnProperty.call(rawCfg, 'trail_smooth_tau_s');
  const headTau = hasHeadTau
    ? clampNumber(cfg.head_smooth_tau_s, DEFAULT_BEV_TRAIL_CONFIG.head_smooth_tau_s)
    : (legacyTau * 0.35);
  const trailTau = hasTrailTau
    ? clampNumber(cfg.trail_smooth_tau_s, DEFAULT_BEV_TRAIL_CONFIG.trail_smooth_tau_s)
    : legacyTau;
  const minStepSceneFloor = Math.max(0.0, clampNumber(cfg.min_step_scene_floor, DEFAULT_BEV_TRAIL_CONFIG.min_step_scene_floor));
  const minStepSceneCeilRaw = Math.max(0.0, clampNumber(cfg.min_step_scene_ceil, DEFAULT_BEV_TRAIL_CONFIG.min_step_scene_ceil));
  const minStepSceneCeil = Math.max(minStepSceneFloor, minStepSceneCeilRaw);
  return {
    enabled: clampBoolean(cfg.enabled, DEFAULT_BEV_TRAIL_CONFIG.enabled),
    window_s: Math.max(0.1, clampNumber(cfg.window_s, DEFAULT_BEV_TRAIL_CONFIG.window_s)),
    draw_stride: Math.max(1, Math.round(clampNumber(cfg.draw_stride, DEFAULT_BEV_TRAIL_CONFIG.draw_stride))),
    min_step_px: Math.max(0.0, clampNumber(cfg.min_step_px, DEFAULT_BEV_TRAIL_CONFIG.min_step_px)),
    min_dt_s: Math.max(0.0, clampNumber(cfg.min_dt_s, DEFAULT_BEV_TRAIL_CONFIG.min_dt_s)),
    smooth_tau_s: legacyTau,
    head_smooth_tau_s: Math.max(0.0, headTau),
    trail_smooth_tau_s: Math.max(0.0, trailTau),
    min_step_scene_floor: minStepSceneFloor,
    min_step_scene_ceil: minStepSceneCeil,
    max_speed_px_per_s: Math.max(0.0, clampNumber(cfg.max_speed_px_per_s, DEFAULT_BEV_TRAIL_CONFIG.max_speed_px_per_s)),
    max_points_per_track: Math.max(2, Math.round(clampNumber(cfg.max_points_per_track, DEFAULT_BEV_TRAIL_CONFIG.max_points_per_track))),
    max_tracks: Math.max(1, Math.round(clampNumber(cfg.max_tracks, DEFAULT_BEV_TRAIL_CONFIG.max_tracks))),
    min_alpha: Math.max(0.0, Math.min(1.0, clampNumber(cfg.min_alpha, DEFAULT_BEV_TRAIL_CONFIG.min_alpha))),
    gap_ms: Math.max(0.0, clampNumber(cfg.gap_ms, DEFAULT_BEV_TRAIL_CONFIG.gap_ms)),
    stale_blink_start_ms: Math.max(0.0, clampNumber(cfg.stale_blink_start_ms, DEFAULT_BEV_TRAIL_CONFIG.stale_blink_start_ms)),
    stale_blink_period_ms: Math.max(1.0, clampNumber(cfg.stale_blink_period_ms, DEFAULT_BEV_TRAIL_CONFIG.stale_blink_period_ms)),
    stale_head_blink_enabled: clampBoolean(cfg.stale_head_blink_enabled, DEFAULT_BEV_TRAIL_CONFIG.stale_head_blink_enabled),
    reset_after_s: Math.max(0.0, clampNumber(cfg.reset_after_s, DEFAULT_BEV_TRAIL_CONFIG.reset_after_s)),
    teleport_break_ratio: Math.max(1.0, clampNumber(cfg.teleport_break_ratio, DEFAULT_BEV_TRAIL_CONFIG.teleport_break_ratio)),
  };
};

export const computeTrailAgeAlpha = (
  nowMs: number,
  sampleMs: number,
  windowMs: number,
  minAlpha: number,
): number => {
  const ageMs = Math.max(0, nowMs - sampleMs);
  const frac = Math.max(0, Math.min(1, 1 - (ageMs / windowMs)));
  return minAlpha + (1 - minAlpha) * frac;
};

export const computeSegmentAlpha = (
  nowMs: number,
  t0: number,
  t1: number,
  windowMs: number,
  minAlpha: number,
): number => computeTrailAgeAlpha(nowMs, (t0 + t1) * 0.5, windowMs, minAlpha);

export const computeSceneUnitsPerPx = (bounds: BoundsLike): number => {
  const spanX = Math.max(1e-6, Number(bounds.xMax) - Number(bounds.xMin));
  const spanZ = Math.max(1e-6, Number(bounds.zMax) - Number(bounds.zMin));
  const widthPx = Math.max(1e-6, Number(bounds.widthPx));
  const heightPx = Math.max(1e-6, Number(bounds.heightPx));
  const unitsPerPxX = spanX / widthPx;
  const unitsPerPxZ = spanZ / heightPx;
  const unitsPerPx = (unitsPerPxX + unitsPerPxZ) * 0.5;
  if (!Number.isFinite(unitsPerPx) || unitsPerPx <= 0) return 1.0;
  return unitsPerPx;
};

export const upsertTrailSample = (track: TrailTrack, args: UpsertTrailSampleArgs): UpsertTrailSampleResult => {
  const { nowMs, doSample, cfg } = args;
  let x = Number(args.x);
  let y = Number(args.y);
  let resetEmaForTeleport = false;
  const sceneUnitsPerPx = Math.max(1e-6, Number(args.sceneUnitsPerPx));
  const minStepSceneRaw = cfg.min_step_px * sceneUnitsPerPx;
  const minStepScene = Math.max(
    cfg.min_step_scene_floor,
    Math.min(cfg.min_step_scene_ceil, minStepSceneRaw)
  );

  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return { headX: x, headY: y, sampled: false };
  }

  pruneTrackWindow(track, nowMs, cfg);

  if (track.lastSeen > 0 && (nowMs - track.lastSeen) > cfg.gap_ms) {
    pushGap(track.points, nowMs);
  }

  if (cfg.reset_after_s > 0.0 && track.lastSeen > 0 && (nowMs - track.lastSeen) > (cfg.reset_after_s * 1000.0)) {
    track.emaX = undefined;
    track.emaY = undefined;
    track.emaTs = undefined;
    track.trailEmaX = undefined;
    track.trailEmaY = undefined;
    track.trailEmaTs = undefined;
  }

  const prevFinite = findLastFinitePoint(track.points);
  if (prevFinite) {
    const prev = prevFinite.point;
    const dtS = Math.max(cfg.min_dt_s, Math.max(0, (nowMs - prev.t) / 1000.0));
    if (dtS > 0 && cfg.max_speed_px_per_s > 0.0) {
      const dx = x - prev.x;
      const dy = y - prev.y;
      const dist = Math.hypot(dx, dy);
      const maxStepSceneRaw = cfg.max_speed_px_per_s * sceneUnitsPerPx * dtS;
      const maxStepScene = Math.max(minStepScene, maxStepSceneRaw);
      if (maxStepScene > 0 && dist > maxStepScene) {
        if (dist > (maxStepScene * cfg.teleport_break_ratio)) {
          pushGap(track.points, nowMs);
          resetEmaForTeleport = true;
        } else {
          const scale = maxStepScene / dist;
          x = prev.x + (dx * scale);
          y = prev.y + (dy * scale);
        }
      }
    }
  }

  let headX = x;
  let headY = y;
  if (cfg.head_smooth_tau_s > 0.0) {
    if (
      resetEmaForTeleport ||
      !Number.isFinite(track.emaX) ||
      !Number.isFinite(track.emaY) ||
      !Number.isFinite(track.emaTs)
    ) {
      track.emaX = headX;
      track.emaY = headY;
      track.emaTs = nowMs;
    } else {
      const dtEmaS = Math.max(0, (nowMs - Number(track.emaTs)) / 1000.0);
      const tau = cfg.head_smooth_tau_s;
      const alpha = (tau > 0.0 && dtEmaS > 0.0) ? (1.0 - Math.exp(-dtEmaS / tau)) : 1.0;
      track.emaX = Number(track.emaX) + alpha * (headX - Number(track.emaX));
      track.emaY = Number(track.emaY) + alpha * (headY - Number(track.emaY));
      track.emaTs = nowMs;
    }
    headX = Number(track.emaX);
    headY = Number(track.emaY);
  } else {
    track.emaX = headX;
    track.emaY = headY;
    track.emaTs = nowMs;
  }

  let trailX = x;
  let trailY = y;
  if (cfg.trail_smooth_tau_s > 0.0) {
    if (
      resetEmaForTeleport ||
      !Number.isFinite(track.trailEmaX) ||
      !Number.isFinite(track.trailEmaY) ||
      !Number.isFinite(track.trailEmaTs)
    ) {
      track.trailEmaX = trailX;
      track.trailEmaY = trailY;
      track.trailEmaTs = nowMs;
    } else {
      const dtTrailS = Math.max(0, (nowMs - Number(track.trailEmaTs)) / 1000.0);
      const tau = cfg.trail_smooth_tau_s;
      const alpha = (tau > 0.0 && dtTrailS > 0.0) ? (1.0 - Math.exp(-dtTrailS / tau)) : 1.0;
      track.trailEmaX = Number(track.trailEmaX) + alpha * (trailX - Number(track.trailEmaX));
      track.trailEmaY = Number(track.trailEmaY) + alpha * (trailY - Number(track.trailEmaY));
      track.trailEmaTs = nowMs;
    }
    trailX = Number(track.trailEmaX);
    trailY = Number(track.trailEmaY);
  } else {
    track.trailEmaX = trailX;
    track.trailEmaY = trailY;
    track.trailEmaTs = nowMs;
  }

  track.lastSeen = nowMs;

  if (!doSample) {
    pruneTrackWindow(track, nowMs, cfg);
    return { headX, headY, sampled: false };
  }

  // Keep BEV trail sampling responsive even when server pushes larger min_dt values.
  const minDtMs = Math.min(50.0, cfg.min_dt_s * 1000.0);
  const lastFinite = findLastFinitePoint(track.points);
  if (lastFinite) {
    const prev = lastFinite.point;
    const dtMs = Math.max(0, nowMs - prev.t);
    const dist = Math.hypot(trailX - prev.x, trailY - prev.y);
    if (dtMs < minDtMs) {
      if (dist >= minStepScene) {
        track.points[lastFinite.index] = { x: trailX, y: trailY, t: prev.t };
        pruneTrackWindow(track, nowMs, cfg);
        return { headX, headY, sampled: true };
      }
      pruneTrackWindow(track, nowMs, cfg);
      return { headX, headY, sampled: false };
    }
    if (dist < minStepScene) {
      pruneTrackWindow(track, nowMs, cfg);
      return { headX, headY, sampled: false };
    }
  }

  track.points.push({ x: trailX, y: trailY, t: nowMs });
  pruneTrackWindow(track, nowMs, cfg);
  return { headX, headY, sampled: true };
};

export const pruneTrailCollection = (
  tracks: Map<string, TrailTrack>,
  nowMs: number,
  cfg: BevTrailConfig
): void => {
  const windowMs = Math.max(100, cfg.window_s * 1000);

  for (const [key, track] of tracks.entries()) {
    pruneTrackWindow(track, nowMs, cfg);
    if (!track.points.length && (nowMs - track.lastSeen) > windowMs) {
      tracks.delete(key);
    }
  }

  if (tracks.size > cfg.max_tracks) {
    const ordered = Array.from(tracks.entries()).sort((a, b) => a[1].lastSeen - b[1].lastSeen);
    for (let i = 0; i < ordered.length - cfg.max_tracks; i += 1) {
      tracks.delete(ordered[i][0]);
    }
  }
};
