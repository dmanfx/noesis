import { useEffect, useRef, useState } from 'react';
import { CameraKey, detectCameraKey } from '../lib/camera';

type Track = { track_id: number; camera_id: string; zone?: string; center?: [number, number]; dwell_time?: number; velocity?: [number, number] };

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
};

export function useWebSocketClient(url: string, handlers: FrameHandlers) {
  const socketRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting');
  const [retry, setRetry] = useState(0);
  const maxRetries = 10;

  useEffect(() => {
    let stop = false;
    const connect = () => {
      setStatus('connecting');
      const ws = new WebSocket(url);
      socketRef.current = ws;
      ws.onopen = () => setStatus('open');
      ws.onclose = () => {
        setStatus('closed');
        if (!stop && retry < maxRetries) {
          setTimeout(() => setRetry(r => r + 1), 5000);
        }
      };
      ws.onerror = () => setStatus('error');
      ws.onmessage = async (ev: MessageEvent) => {
        try {
          if (ev.data instanceof Blob) {
            const arrayBuffer = await ev.data.arrayBuffer();
            if (arrayBuffer.byteLength < 1) return;
            const view = new DataView(arrayBuffer);
            const idLen = view.getUint8(0);
            if (arrayBuffer.byteLength < 1 + idLen) return;
            const idBytes = new Uint8Array(arrayBuffer, 1, idLen);
            const id = new TextDecoder('utf-8').decode(idBytes);
            const jpeg = arrayBuffer.slice(1 + idLen);
            const blob = new Blob([jpeg], { type: 'image/jpeg' });
            const cam = detectCameraKey(id);
            if (cam) handlers.onImage(cam, blob);
            return;
          }
          if (ev.data instanceof ArrayBuffer) {
            const blob = new Blob([ev.data], { type: 'application/octet-stream' });
            const arrayBuffer = await blob.arrayBuffer();
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
          if (data.type === 'stats' && data.payload) {
            handlers.onStats(data.payload as StatsPayload);
          } else if (data.type === 'toggle_update' && data.toggle_name === 'trail_visualization_enabled') {
            handlers.onTrailToggle?.(!!data.enabled);
          } else if (data.type === 'trail_visualization_enabled_update') {
            handlers.onTrailToggle?.(!!data.enabled);
          }
        } catch (e) {
          // swallow parsing errors
        }
      };
    };
    connect();
    return () => { stop = true; socketRef.current?.close(); };
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
    sendDetectionToggle: (name: string, enabled: boolean) => sendJson({ type: 'set_detection_toggle', toggle_name: name, enabled })
  };
}

