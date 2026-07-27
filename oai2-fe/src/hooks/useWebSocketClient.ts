import { useEffect, useRef, useState } from 'react';
import { setCalibration } from '../lib/calibration';
import { CameraKey, detectCameraKey } from '../lib/camera';
import { wsLog } from '../lib/wsLogger';
import { LatencyMetrics } from '../types/latency';

type Track = {
  stable_id: number;
  tracker_id?: number;
  id_display?: string;
  display_name?: string;
  identity_kind?: string;
  identity_state?: string;
  resident_uuid?: string;
  camera_id: string;
  zone?: string;
  center?: [number, number];
  dwell_time?: number;
  velocity?: [number, number];
  world?: [number, number, number];
  world_valid?: boolean;
  world_frame?: string;
  world_source?: string;
  world_quality?: string;
  world_quality_reason?: string;
  depth_status?: string | null;
  depth_anchor_source?: string | null;
  depth_anchor_m?: number | null;
  depth_used_m?: number | null;
  depth_center_m?: number | null;
  depth_median_m?: number | null;
  depth_sample_count?: number | null;
  depth_valid_fraction?: number | null;
};

export type CamerasStats = Record<string, {
  status?: string;
  frame_count?: number;
  latency_ms?: LatencyMetrics;
  tracking?: {
    occupancy?: Record<string, number>;
    active_tracks?: Track[];
    transitions?: Array<{ timestamp?: number; stable_id?: number; from_zone?: string; to_zone?: string; camera_id?: string; line_name?: string }>;
  }
}>;

export type MosaicLayout = {
  mosaic_w?: number;
  mosaic_h?: number;
  rows?: number | null;
  cols?: number | null;
  source_count?: number;
  sources?: Array<{ source_id: number; camera_id: string }>;
  frame_w?: number;
  frame_h?: number;
  square_seq_grid?: boolean;
  tile_order?: string;
};

export interface StatsPayload {
  uptime?: number;
  cameras?: CamerasStats;
  pipeline?: {
    analytics_reload_count?: number;
    mosaic_layout?: MosaicLayout;
    latency_ms?: LatencyMetrics;
  };
}

export type FrameHandlers = {
  onBevMeta?: (payload: any) => void;
  onStats: (stats: StatsPayload) => void;
  onTrailToggle?: (enabled: boolean) => void;
  onTrailSettings?: (config: Record<string, unknown>) => void;
  onCalibration?: (bundle: any) => void;
  onMADiagnostics?: (payload: any) => void;
  onMADepth?: (
    payload: any,
    context?: {
      requestId: string;
      deadlineAtMs: number;
      expectedCameraId: string;
    },
  ) => void | Promise<void>;
  onFloorplan?: (payload: any) => void;
  onAutoCalibrateResult?: (payload: any) => void;
  // WebRTC signaling handlers
  onWebRTCAnswer?: (sdp: string) => void;
  onWebRTCIceCandidate?: (candidate: RTCIceCandidateInit) => void;
  onWebRTCError?: (error: string) => void;
};

export type FloorplanRequest = {
  camera?: string;
  maxAgeSec?: number;
  gridResM?: number;
  maxExtentM?: number;
  requestId?: string;
  cacheOnly?: boolean;
  snapshotRef?: string;
  snapshotId?: string;
  snapshotContentSha256?: string;
};

export type DepthRequestStrategy = 'fresh' | 'cache-first' | 'cache-only';

const DEPTH_REQUEST_TIMEOUT_MS = 150_000;

type PendingDepthRequest = {
  cameraId: string;
  deadlineAtMs: number;
  timeout: ReturnType<typeof setTimeout>;
};

export function useWebSocketClient(url: string, handlers: FrameHandlers) {
  const socketRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<'connecting' | 'open' | 'closed' | 'error'>('connecting');
  const [retry, setRetry] = useState(0);
  const maxRetries = 10;
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const depthInFlightRef = useRef<Record<string, Set<string>>>({});
  const depthRequestTimeoutsRef = useRef<Map<string, PendingDepthRequest>>(new Map());
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  const finishDepthRequest = (cameraId?: unknown, requestId?: unknown) => {
    const requestKey = requestId === undefined || requestId === null
      ? ''
      : String(requestId);
    if (requestKey) {
      const request = depthRequestTimeoutsRef.current.get(requestKey);
      if (request) clearTimeout(request.timeout);
      depthRequestTimeoutsRef.current.delete(requestKey);
      for (const [camera, pending] of Object.entries(depthInFlightRef.current)) {
        if (!pending.delete(requestKey)) continue;
        if (pending.size === 0) delete depthInFlightRef.current[camera];
        return;
      }
    }
    const cameraKey = cameraId === undefined || cameraId === null
      ? ''
      : String(cameraId);
    const pending = depthInFlightRef.current[cameraKey];
    if (!pending) return;
    if (requestKey) pending.delete(requestKey);
    if (!requestKey || pending.size === 0) {
      delete depthInFlightRef.current[cameraKey];
    }
  };

  const clearAllDepthRequests = () => {
    for (const request of depthRequestTimeoutsRef.current.values()) {
      clearTimeout(request.timeout);
    }
    depthRequestTimeoutsRef.current.clear();
    depthInFlightRef.current = {};
  };

  const registerDepthRequest = (cameraId: string, requestId: string) => {
    if (!cameraId || !requestId) return;
    const bucket = depthInFlightRef.current[cameraId] || new Set<string>();
    bucket.add(requestId);
    depthInFlightRef.current[cameraId] = bucket;
    const deadlineAtMs = performance.now() + DEPTH_REQUEST_TIMEOUT_MS;
    const timeout = setTimeout(
      () => finishDepthRequest(cameraId, requestId),
      DEPTH_REQUEST_TIMEOUT_MS,
    );
    depthRequestTimeoutsRef.current.set(requestId, {
      cameraId,
      deadlineAtMs,
      timeout,
    });
  };

  const summarizeWsMessage = (obj: any): Record<string, unknown> => {
    if (!obj || typeof obj !== 'object') return {};
    const type = (obj as any).type;
    const base: Record<string, unknown> = {
      type,
      request_id: (obj as any).request_id ?? (obj as any).requestId,
      camera: (obj as any).camera ?? (obj as any).cam_id ?? (obj as any).camera_id ?? (obj as any).cameraId,
    };
    if (type === 'get_floorplan') {
      return {
        ...base,
        max_age_sec: (obj as any).max_age_sec,
        grid_res_m: (obj as any).grid_res_m,
        max_extent_m: (obj as any).max_extent_m,
        cache_only: (obj as any).cache_only,
        snapshot_ref: (obj as any).snapshot_ref,
        snapshot_id: (obj as any).snapshot_id,
        snapshot_content_sha256: (obj as any).snapshot_content_sha256,
      };
    }
    if (type === 'get_ma_depth' || type === 'get_ma_depth_cache') {
      return {
        ...base,
        ts_max_us: (obj as any).ts_max_us,
        cache_only: (obj as any).cache_only,
      };
    }
    return base;
  };

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
            wsLog.warn('[WS] Failed to send heartbeat:', e);
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
      try { ws.binaryType = 'arraybuffer'; } catch { }
      socketRef.current = ws;

      ws.onopen = () => {
        if (socketRef.current !== ws) {
          ws.close();
          return;
        }
        setStatus('open');
        startHeartbeat(ws);
        wsLog.info('[WS] Connected (heartbeat started)');
      };

      ws.onclose = (event) => {
        if (socketRef.current !== ws) return;
        setStatus('closed');
        stopHeartbeat();
        clearAllDepthRequests();
        wsLog.info('[WS] Closed', { code: event.code, reason: event.reason });

        if (!stop && retry < maxRetries) {
          const delay = Math.min(5000 * Math.pow(2, retry), 30000); // Exponential backoff, max 30s
          wsLog.info('[WS] Reconnect scheduled', { attempt: retry + 1, maxRetries, delayMs: delay });
          reconnectTimeoutRef.current = setTimeout(() => setRetry(r => r + 1), delay);
        }
      };

      ws.onerror = (error) => {
        if (socketRef.current !== ws) return;
        setStatus('error');
        wsLog.error('[WS] Error:', error);
      };
      ws.onmessage = async (ev: MessageEvent) => {
        if (socketRef.current !== ws) return;
        try {
          if (ev.data instanceof ArrayBuffer || ev.data instanceof Blob) {
            // BEV JPEG binary delivery retired (meta-only mode). The framed `bev:<cam>` binary path is no longer produced
            // by the canonical DS8 BevRenderer. Future binary depth (if implemented) would use a different magic/header.
            // Silently ignore for now (or log at debug if needed).
            return;
          }
          const data = JSON.parse(ev.data);

          // Handle ping messages by responding with pong
          if (data.type === 'ping') {
            try {
              ws.send(JSON.stringify({ type: 'pong', timestamp: data.timestamp }));
              wsLog.debug('[WS] Sent pong response to server ping');
            } catch (e) {
              wsLog.warn('[WS] Failed to send pong response:', e);
            }
            return;
          }

          if (data.type === 'auto_calibrate_result') {
            wsLog.debug('[WS] auto-calibrate result', data);
            try { handlersRef.current.onAutoCalibrateResult?.(data); } catch { }
            return;
          }

          // Handle pong responses from server
          if (data.type === 'pong') {
            const latency = Date.now() - (data.timestamp || 0);
            wsLog.debug('[WS] Received pong', { latencyMs: latency });
            return;
          }

          if (data.type === 'stats' && data.payload) {
            handlersRef.current.onStats(data.payload as StatsPayload);
          } else if (data.type === 'calibration-bundle' && data.data) {
            try { setCalibration(data); } catch { }
            try {
              handlersRef.current.onCalibration?.(data.data);
            } catch (err) {
              console.warn('Calibration handler failed', err);
            }
          } else if (data.type === 'toggle_update' && data.toggle_name === 'trail_visualization_enabled') {
            handlersRef.current.onTrailToggle?.(!!data.enabled);
          } else if (data.type === 'trail_visualization_enabled_update') {
            handlersRef.current.onTrailToggle?.(!!data.enabled);
          } else if (data.type === 'trail_settings_update') {
            const cfg = (data && typeof data.config === 'object' && data.config !== null)
              ? data.config as Record<string, unknown>
              : null;
            if (cfg) handlersRef.current.onTrailSettings?.(cfg);
          } else if (data.type === 'ma_diagnostics') {
            handlersRef.current.onMADiagnostics?.(data);
          } else if (data.type === 'ma_depth_response') {
            const cam = data.camera || data.cam_id || data.cameraId || data.camera_id || data.camId;
            const requestId = data.request_id || data.requestId || data.requestID;
            const requestKey = requestId === undefined || requestId === null
              ? ''
              : String(requestId);
            const pendingRequest = requestKey
              ? depthRequestTimeoutsRef.current.get(requestKey)
              : undefined;
            if (
              !requestKey
              || !pendingRequest
              || performance.now() >= pendingRequest.deadlineAtMs
            ) {
              finishDepthRequest(cam, requestKey);
              wsLog.warn('[WS] late or unknown depth response rejected', {
                camera: String(cam || ''),
                requestId: requestKey,
              });
              return;
            }
            try {
              const completion = handlersRef.current.onMADepth?.(data, {
                requestId: requestKey,
                deadlineAtMs: pendingRequest.deadlineAtMs,
                expectedCameraId: pendingRequest.cameraId,
              });
              if (completion && typeof (completion as Promise<void>).then === 'function') {
                void Promise.resolve(completion)
                  .catch((error) => wsLog.warn('[WS] depth handler failed', { error: String(error) }))
                  .finally(() => finishDepthRequest(cam, requestKey));
              } else {
                finishDepthRequest(cam, requestKey);
              }
            } catch (error) {
              finishDepthRequest(cam, requestKey);
              throw error;
            }
          } else if (data.type === 'floorplan_response') {
            handlersRef.current.onFloorplan?.(data);
          } else if (data.type === 'bev-frame') {
            handlersRef.current.onBevMeta?.(data);
          } else if (data.type === 'bev-status') {
            handlersRef.current.onBevMeta?.(data);
          }
          // WebRTC signaling responses from server
          else if (data.type === 'webrtc_answer' && data.sdp) {
            handlersRef.current.onWebRTCAnswer?.(data.sdp);
          } else if (data.type === 'webrtc_ice_candidate' && data.candidate) {
            handlersRef.current.onWebRTCIceCandidate?.({
              candidate: data.candidate,
              sdpMLineIndex: data.sdpMLineIndex ?? 0,
              sdpMid: data.sdpMid,
            });
          } else if (data.type === 'webrtc_error') {
            handlersRef.current.onWebRTCError?.(data.error || 'Unknown WebRTC error');
          }
        } catch (e) {
          // Previously swallowed silently — major source of "BEV went blank with no signal".
          // Log the message type (if any) + short preview so BEV feed problems become visible in the console/wsLog.
          try {
            const preview = typeof ev.data === 'string' ? ev.data.slice(0, 200) : '[binary]';
            wsLog.warn('[WS] onmessage handler error (was previously swallowed)', {
              error: String(e),
              preview,
            });
          } catch {}
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
      clearAllDepthRequests();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, retry]);

  const sendJson = (obj: any) => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    if (obj && (obj.type === 'get_ma_depth' || obj.type === 'get_ma_depth_cache' || obj.type === 'get_floorplan')) {
      const summary = summarizeWsMessage(obj);
      const key = `send:${String(obj.type)}:${String((summary as any).camera ?? '')}`;
      wsLog.debugRateLimited(
        key,
        ['[WS] send', String(obj.type), summary],
        () => {
          wsLog.debug('payload', { ...obj, img_b64: undefined });
        }
      );
    }
    ws.send(JSON.stringify(obj));
    return true;
  };

  const normalizeTsUs = (value?: number): number | undefined => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return undefined;
    if (value < 1_000_000_000) return Math.floor(value * 1_000_000);
    if (value < 1_000_000_000_000) return Math.floor(value * 1000);
    return Math.floor(value);
  };

  const computeTsMaxUs = (strategy: DepthRequestStrategy, override?: number): number | undefined => {
    const normalized = normalizeTsUs(override);
    if (typeof normalized === 'number') {
      return normalized;
    }
    if (strategy === 'cache-first' || strategy === 'cache-only') {
      return Math.floor(Date.now() * 1000);
    }
    return undefined;
  };

  return {
    status,
    sendClearStats: () => sendJson({ type: 'clear_stats' }),
    sendTrailToggle: (enabled: boolean) => sendJson({ type: 'set_vis_toggle', toggle_name: 'trail_visualization_enabled', enabled }),
    sendAutoCalibrate: (camId?: string) => sendJson({ type: 'auto_calibrate_pose', camera: camId }),
    requestMapAnythingDepth: (camId: string, strategy: DepthRequestStrategy = 'fresh', tsMaxOverride?: number) => {
      if (!camId) return false;
      const existing = depthInFlightRef.current[camId];
      // Keep a camera's snapshot chain single-flight. In particular, repeated
      // refresh clicks must not start competing fresh inferences whose
      // floorplan continuations could arrive out of order.
      if (existing && existing.size > 0) {
        return false;
      }
      const requestId = `${camId}-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
      const tsMaxUs = computeTsMaxUs(strategy, tsMaxOverride);
      const payload: Record<string, unknown> = {
        type: strategy === 'cache-only' ? 'get_ma_depth_cache' : 'get_ma_depth',
        camera: camId,
        request_id: requestId
      };
      if (typeof tsMaxUs === 'number') {
        payload.ts_max_us = tsMaxUs;
      }
      if (strategy === 'cache-only') {
        payload.cache_only = true;
      }
      const ok = sendJson(payload);
      if (ok) {
        registerDepthRequest(camId, requestId);
      }
      return ok;
    },
    requestFloorplan: (options?: FloorplanRequest) => {
      const requestId = options?.requestId || Date.now().toString();
      const payload = {
        type: 'get_floorplan',
        request_id: requestId,
        camera: options?.camera,
        // For regenerate flows, callers should pass maxAgeSec=0 to ignore staleness
        max_age_sec: options?.maxAgeSec ?? 60,
        grid_res_m: options?.gridResM ?? 0.04,
        max_extent_m: options?.maxExtentM ?? 20,
        // Omitted options must never start inference. A fresh floorplan request
        // is always explicit and, for the depth panel, bound to one snapshot.
        cache_only: options?.cacheOnly ?? true,
        ...(options?.snapshotRef ? { snapshot_ref: options.snapshotRef } : {}),
        ...(options?.snapshotId ? { snapshot_id: options.snapshotId } : {}),
        ...(options?.snapshotContentSha256
          ? { snapshot_content_sha256: options.snapshotContentSha256 }
          : {}),
      };
      const ok = sendJson(payload);
      return ok ? requestId : '';
    },
    sendBevConfig: (camId: string, config: any) => sendJson({ type: 'bev-config', cameraId: camId, config }),
    sendBevOverlay: (camId: string, enabled: boolean) => sendJson({ type: 'bev-overlay', cameraId: camId, enabled }),
    // WebRTC signaling senders
    sendWebRTCOffer: (sdp: string) => sendJson({ type: 'webrtc_offer', sdp }),
    sendWebRTCIceCandidate: (candidate: RTCIceCandidateInit) => sendJson({
      type: 'webrtc_ice_candidate',
      candidate: candidate.candidate,
      sdpMLineIndex: candidate.sdpMLineIndex,
      sdpMid: candidate.sdpMid,
    }),
  };
}
