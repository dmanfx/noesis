export const NOESIS_REST_BASE = '';
export const NOESIS_WEBSOCKET_PATH = '/api/noesis/ws';

type BrowserLocation = Pick<Location, 'origin' | 'protocol'>;

export function noesisWebSocketUrl(location: BrowserLocation = window.location): string {
  const url = new URL(NOESIS_WEBSOCKET_PATH, location.origin);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}
