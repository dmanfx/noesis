type WsLogLevel = 'silent' | 'error' | 'warn' | 'info' | 'debug';

type WsLogConfig = {
  level: WsLogLevel;
  rateLimitMs: number;
};

const LEVEL_ORDER: Record<Exclude<WsLogLevel, 'silent'>, number> = {
  error: 0,
  warn: 1,
  info: 2,
  debug: 3,
};

const DEFAULT_CONFIG: WsLogConfig = {
  level: 'info',
  rateLimitMs: 2000,
};

const rateState = new Map<string, { lastEmitMs: number; suppressed: number }>();

const readLocalStorage = (key: string): string | null => {
  try {
    return globalThis.localStorage?.getItem(key) ?? null;
  } catch {
    return null;
  }
};

const readQueryParam = (key: string): string | null => {
  try {
    const search = globalThis.location?.search;
    if (!search) return null;
    return new URLSearchParams(search).get(key);
  } catch {
    return null;
  }
};

const parseWsLogLevel = (raw: unknown): WsLogLevel | null => {
  if (typeof raw !== 'string') return null;
  const v = raw.trim().toLowerCase();
  if (v === 'silent') return 'silent';
  if (v === 'error') return 'error';
  if (v === 'warn' || v === 'warning') return 'warn';
  if (v === 'info') return 'info';
  if (v === 'debug') return 'debug';
  return null;
};

const parseRateLimitMs = (raw: unknown): number | null => {
  if (typeof raw !== 'string') return null;
  const n = Number(raw);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.floor(n));
};

export const getWsLogConfig = (): WsLogConfig => {
  const levelFromLs = parseWsLogLevel(readLocalStorage('WS_LOG_LEVEL'));
  const levelFromParam = parseWsLogLevel(readQueryParam('wsLog'));

  const debugParam = readQueryParam('debug');
  const debugIsWs =
    typeof debugParam === 'string' &&
    debugParam
      .split(',')
      .map(s => s.trim().toLowerCase())
      .some(v => v === 'ws' || v === 'websocket');

  const level = levelFromParam ?? levelFromLs ?? (debugIsWs ? 'debug' : DEFAULT_CONFIG.level);

  const rateFromLs = parseRateLimitMs(readLocalStorage('WS_LOG_RATE_MS'));
  const rateFromParam = parseRateLimitMs(readQueryParam('wsLogRateMs'));
  const rateLimitMs = rateFromParam ?? rateFromLs ?? DEFAULT_CONFIG.rateLimitMs;

  return { level, rateLimitMs };
};

const levelEnabled = (config: WsLogConfig, desired: Exclude<WsLogLevel, 'silent'>): boolean => {
  if (config.level === 'silent') return false;
  return LEVEL_ORDER[desired] <= LEVEL_ORDER[config.level];
};

export const wsLog = {
  error: (...args: unknown[]) => {
    const cfg = getWsLogConfig();
    if (!levelEnabled(cfg, 'error')) return;
    // eslint-disable-next-line no-console
    console.error(...args);
  },
  warn: (...args: unknown[]) => {
    const cfg = getWsLogConfig();
    if (!levelEnabled(cfg, 'warn')) return;
    // eslint-disable-next-line no-console
    console.warn(...args);
  },
  info: (...args: unknown[]) => {
    const cfg = getWsLogConfig();
    if (!levelEnabled(cfg, 'info')) return;
    // eslint-disable-next-line no-console
    console.log(...args);
  },
  debug: (...args: unknown[]) => {
    const cfg = getWsLogConfig();
    if (!levelEnabled(cfg, 'debug')) return;
    // eslint-disable-next-line no-console
    console.debug(...args);
  },
  debugRateLimited: (key: string, headerArgs: unknown[], groupBody?: () => void) => {
    const cfg = getWsLogConfig();
    if (!levelEnabled(cfg, 'debug')) return;

    const now = Date.now();
    const state = rateState.get(key) ?? { lastEmitMs: 0, suppressed: 0 };
    const withinWindow = cfg.rateLimitMs > 0 && now - state.lastEmitMs < cfg.rateLimitMs;

    if (withinWindow) {
      state.suppressed += 1;
      rateState.set(key, state);
      return;
    }

    const suppressed = state.suppressed;
    state.lastEmitMs = now;
    state.suppressed = 0;
    rateState.set(key, state);

    const finalHeaderArgs = suppressed > 0 ? [...headerArgs, { suppressed }] : headerArgs;

    if (groupBody) {
      // eslint-disable-next-line no-console
      console.groupCollapsed(...finalHeaderArgs);
      try {
        groupBody();
      } finally {
        // eslint-disable-next-line no-console
        console.groupEnd();
      }
      return;
    }

    // eslint-disable-next-line no-console
    console.debug(...finalHeaderArgs);
  },
};

