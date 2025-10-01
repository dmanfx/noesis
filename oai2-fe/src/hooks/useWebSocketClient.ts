import { useEffect, useRef, useState } from 'react';
import { setCalibration } from '../lib/calibration';
import { CameraKey, detectCameraKey } from '../lib/camera';

type Track = {
  track_id: number;
  stable_id?: number | null;
  camera_id: string;
  zone?: string;
  center?: [number, number];
  dwell_time?: number;
  velocity?: [number, number];
};

export type CamerasStats = Record<string, {
  status?: string;
  frame_count?: number;
  tracking?: {
    occupancy?: Record<string, number>;
    active_tracks?: Track[];
    transitions?: Array<{ timestamp?: number; track_id?: number; from_zone?: string; to_zone?: string; camera_id?: string }>;
  }
}>;

export interface StatsPayload {
  uptime?: number;
  cameras?: CamerasStats;
}

export type FrameHandlers = {
  onImage: (cam: CameraKey, blob: Blob) => void;
  onStats: (stats: StatsPayload) => void;
  onTrailToggle?: (enabled: boolean) => void;
  onMADiagnostics?: (payload: any) => void;
  onMADepth?: (payload: any) => void;
  onFloorplan?: (payload: any) => void;
};

export type FloorplanRequest = {
  cameras?: string[];
  maxAgeSec?: number;
  gridResM?: number;
  maxExtentM?: number;
  useHeight?: boolean;
  requestId?: string;
};

export function useWebSocketClient(url: string, handlers: FrameHandlers) {
  const socketRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting');
  const [retry, setRetry] = useState(0);
  const maxRetries = 10;
  const heartbeatRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    let stop = false;

    const startHeartbeat = (ws: WebSocket) => {
      // Clear any existing heartbeat
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
      }

      // Send a heartbeat every 30 seconds to keep connection alive
      heartbeatRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          try {
            // Send a simple ping message that the server will echo back
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          } catch (e) {
            console.warn('Failed to send heartbeat:', e);
          }
        }
      }, 30000); // 30 seconds
    };

    const stopHeartbeat = () => {
      if (heartbeatRef.current) {
        clearInterval(heartbeatRef.current);
        heartbeatRef.current = null;
      }
    };

    const connect = () => {
      setStatus('connecting');

      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      const ws = new WebSocket(url);
      // Prefer ArrayBuffer to avoid extra Blob conversions
      try { ws.binaryType = 'arraybuffer'; } catch {}
      socketRef.current = ws;

      ws.onopen = () => {
        setStatus('open');
        startHeartbeat(ws);
        console.log('WebSocket connected, heartbeat started');
      };

      ws.onclose = (event) => {
        setStatus('closed');
        stopHeartbeat();
        console.log(`WebSocket closed: code=${event.code}, reason=${event.reason}`);

        if (!stop && retry < maxRetries) {
          const delay = Math.min(5000 * Math.pow(2, retry), 30000); // Exponential backoff, max 30s
          console.log(`Attempting reconnection ${retry + 1}/${maxRetries} in ${delay}ms`);
          reconnectTimeoutRef.current = setTimeout(() => setRetry(r => r + 1), delay);
        }
      };

      ws.onerror = (error) => {
        setStatus('error');
        console.error('WebSocket error:', error);
      };
      ws.onmessage = async (ev: MessageEvent) => {
        try {
          if (ev.data instanceof ArrayBuffer) {
            const arrayBuffer = ev.data as ArrayBuffer;
            const view = new DataView(arrayBuffer);
            const idLen = view.getUint8(0);
            if (arrayBuffer.byteLength < 1 + idLen) return;
            const idBytes = new Uint8Array(arrayBuffer, 1, idLen);
            const id = new TextDecoder('utf-8').decode(idBytes);
            const jpeg = arrayBuffer.slice(1 + idLen);
            const jpgBlob = new Blob([jpeg], { type: 'image/jpeg' });
            const cam = detectCameraKey(id);
            if (cam) handlers.onImage(cam, jpgBlob);
            return;
          }
          if (ev.data instanceof Blob) {
            // Fallback path when proxies force Blob delivery
            const arrayBuffer = await (ev.data as Blob).arrayBuffer();
            if (arrayBuffer.byteLength < 1) return;
            const view = new DataView(arrayBuffer);
            const idLen = view.getUint8(0);
            if (arrayBuffer.byteLength < 1 + idLen) return;
            const idBytes = new Uint8Array(arrayBuffer, 1, idLen);
            const id = new TextDecoder('utf-8').decode(idBytes);
            const jpeg = arrayBuffer.slice(1 + idLen);
            const jpgBlob = new Blob([jpeg], { type: 'image/jpeg' });
            const cam = detectCameraKey(id);
            if (cam) handlers.onImage(cam, jpgBlob);
            return;
          }
          const data = JSON.parse(ev.data);

          // Handle ping messages by responding with pong
          if (data.type === 'ping') {
            try {
              ws.send(JSON.stringify({ type: 'pong', timestamp: data.timestamp }));
              console.debug('Sent pong response to server ping');
            } catch (e) {
              console.warn('Failed to send pong response:', e);
            }
            return;
          }

          // Handle pong responses from server
          if (data.type === 'pong') {
            const latency = Date.now() - (data.timestamp || 0);
            console.debug(`Received pong from server (latency: ${latency}ms)`);
            return;
          }

          if (data.type === 'stats' && data.payload) {
            handlers.onStats(data.payload as StatsPayload);
          } else if (data.type === 'calibration-bundle' && data.data) {
            try { setCalibration(data); } catch {}
          } else if (data.type === 'toggle_update' && data.toggle_name === 'trail_visualization_enabled') {
            handlers.onTrailToggle?.(!!data.enabled);
          } else if (data.type === 'trail_visualization_enabled_update') {
            handlers.onTrailToggle?.(!!data.enabled);
          } else if (data.type === 'ma_diagnostics') {
            handlers.onMADiagnostics?.(data);
          } else if (data.type === 'ma_depth_response') {
            handlers.onMADepth?.(data);
          } else if (data.type === 'floorplan_response') {
            handlers.onFloorplan?.(data);
          }
        } catch (e) {
          // swallow parsing errors
        }
      };
    };
    connect();
    return () => {
      stop = true;
      stopHeartbeat();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      socketRef.current?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, retry]);

  const sendJson = (obj: any) => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(obj));
    return true;
  };

  return {
    status,
    sendClearStats: () => sendJson({ type: 'clear_stats' }),
    sendTrailToggle: (enabled: boolean) => sendJson({ type: 'set_vis_toggle', toggle_name: 'trail_visualization_enabled', enabled }),
    sendDetectionConfig: (config: any) => sendJson({ type: 'update_detection_config', config }),
    sendDetectionToggle: (name: string, enabled: boolean) => sendJson({ type: 'set_detection_toggle', toggle_name: name, enabled }),
    requestMapAnythingDepth: (camId: string, tsMax?: number) => sendJson({ type: 'get_ma_depth', camId, ts_max: tsMax ?? Date.now() }),
    requestFloorplan: (options?: FloorplanRequest) => {
      const requestId = options?.requestId || Date.now().toString();
      const ok = sendJson({
        type: 'get_floorplan',
        request_id: requestId,
        cameras: options?.cameras,
        max_age_sec: options?.maxAgeSec ?? 60,
        grid_res_m: options?.gridResM ?? 0.5,
        max_extent_m: options?.maxExtentM ?? 20,
        use_height: options?.useHeight ?? false
      });
      return ok ? requestId : '';
    }
  };
}
