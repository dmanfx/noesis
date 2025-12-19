import { useEffect, useRef, useState, useCallback } from 'react';
import type React from 'react';

/**
 * Configuration for the WebRTC client
 */
interface WebRTCConfig {
    /** STUN server URLs for ICE candidates */
    stunServers?: string[];
    /** Enable/disable verbose logging */
    debug?: boolean;
    /** Optional callback for reporting inbound video FPS */
    onVideoFps?: (fps: number) => void;
    /** Interval for polling WebRTC stats (ms) */
    statsIntervalMs?: number;
}

/**
 * WebRTC signaling functions passed from useWebSocketClient
 */
interface WebRTCSignaling {
    /** Send WebRTC offer SDP to server */
    sendOffer: (sdp: string) => void;
    /** Send ICE candidate to server */
    sendIceCandidate: (candidate: RTCIceCandidateInit) => void;
}

/**
 * Return type for useWebRTCClient hook
 */
interface UseWebRTCClientReturn {
    /** Ref to attach to <video> element */
    videoRef: React.RefObject<HTMLVideoElement>;
    /** Current connection state */
    connectionState: RTCPeerConnectionState | 'new';
    /** Whether stream is connected and playing */
    isConnected: boolean;
    /** Start WebRTC connection (sends offer to server) */
    connect: () => Promise<void>;
    /** Close WebRTC connection */
    disconnect: () => void;
    /** Handle incoming WebRTC answer from server */
    handleAnswer: (sdp: string) => Promise<void>;
    /** Handle incoming ICE candidate from server */
    handleIceCandidate: (candidate: RTCIceCandidateInit) => Promise<void>;
}

const DEFAULT_STUN_SERVERS = [
    'stun:stun.l.google.com:19302',
    'stun:stun1.l.google.com:19302',
];

/**
 * React hook for WebRTC peer connection management.
 * 
 * This hook manages the RTCPeerConnection lifecycle and provides methods
 * for WebRTC signaling (offer/answer/ICE) via the existing WebSocket connection.
 * 
 * @param signaling - Functions to send signaling messages via WebSocket
 * @param config - Optional WebRTC configuration
 * @returns WebRTC client controls and state
 */
export function useWebRTCClient(
    signaling: WebRTCSignaling | null,
    config: WebRTCConfig = {}
): UseWebRTCClientReturn {
    const {
        stunServers = DEFAULT_STUN_SERVERS,
        debug = false,
        onVideoFps,
        statsIntervalMs = 1000,
    } = config;
    const wantStats = debug || typeof onVideoFps === 'function';

    const videoRef = useRef<HTMLVideoElement>(null);
    const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
    const remoteStreamRef = useRef<MediaStream | null>(null);
    const statsTimerRef = useRef<number | null>(null);
    const lastVideoStatRef = useRef<{ frames?: number; ts?: number }>({});
    const [connectionState, setConnectionState] = useState<RTCPeerConnectionState | 'new'>('new');
    const [isConnected, setIsConnected] = useState(false);

    const log = useCallback((...args: unknown[]) => {
        if (debug) {
            console.log('[WebRTC]', ...args);
        }
    }, [debug]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (statsTimerRef.current) {
                window.clearInterval(statsTimerRef.current);
                statsTimerRef.current = null;
            }
            if (peerConnectionRef.current) {
                peerConnectionRef.current.close();
                peerConnectionRef.current = null;
            }
        };
    }, []);

    const createPeerConnection = useCallback((): RTCPeerConnection => {
        const pc = new RTCPeerConnection({
            iceServers: stunServers.map(url => ({ urls: url })),
        });

        pc.onicecandidate = (event) => {
            if (event.candidate && signaling) {
                log('Sending ICE candidate:', event.candidate.candidate);
                signaling.sendIceCandidate(event.candidate.toJSON());
            }
        };

        pc.oniceconnectionstatechange = () => {
            log('ICE connection state:', pc.iceConnectionState);
        };

        pc.onconnectionstatechange = () => {
            log('Connection state:', pc.connectionState);
            setConnectionState(pc.connectionState);
            setIsConnected(pc.connectionState === 'connected');

            if (statsTimerRef.current) {
                window.clearInterval(statsTimerRef.current);
                statsTimerRef.current = null;
            }
            if (pc.connectionState !== 'connected') {
                lastVideoStatRef.current = {};
            }

            if (wantStats && pc.connectionState === 'connected') {
                const interval = Math.max(250, statsIntervalMs);
                statsTimerRef.current = window.setInterval(() => {
                    pc.getStats()
                        .then((stats) => {
                            let inbound: any = null;
                            let pair: any = null;
                            stats.forEach((r) => {
                                const anyR: any = r as any;
                                if (anyR.type === 'inbound-rtp' && anyR.kind === 'video') inbound = anyR;
                                if (anyR.type === 'candidate-pair' && anyR.state === 'succeeded' && anyR.nominated) pair = anyR;
                            });
                            if (inbound) {
                                if (debug) {
                                    log('Inbound video stats:', {
                                        bytesReceived: inbound.bytesReceived,
                                        packetsReceived: inbound.packetsReceived,
                                        packetsLost: inbound.packetsLost,
                                        framesDecoded: inbound.framesDecoded,
                                        framesReceived: inbound.framesReceived,
                                        framesDropped: inbound.framesDropped,
                                        jitter: inbound.jitter,
                                        decoderImplementation: inbound.decoderImplementation,
                                        framesPerSecond: inbound.framesPerSecond,
                                    });
                                }

                                if (onVideoFps) {
                                    const nowTs = performance.now();
                                    const frames = typeof inbound.framesDecoded === 'number'
                                        ? inbound.framesDecoded
                                        : (typeof inbound.framesReceived === 'number' ? inbound.framesReceived : null);
                                    let fps: number | null = null;
                                    if (typeof inbound.framesPerSecond === 'number' && Number.isFinite(inbound.framesPerSecond)) {
                                        fps = inbound.framesPerSecond;
                                    } else if (frames !== null) {
                                        const last = lastVideoStatRef.current;
                                        if (typeof last.frames === 'number' && typeof last.ts === 'number' && last.ts > 0) {
                                            const deltaFrames = frames - last.frames;
                                            const deltaMs = nowTs - last.ts;
                                            if (deltaFrames >= 0 && deltaMs > 1) {
                                                fps = (deltaFrames * 1000) / deltaMs;
                                            }
                                        }
                                    }
                                    if (frames !== null) {
                                        lastVideoStatRef.current = { frames, ts: nowTs };
                                    }
                                    if (fps !== null && Number.isFinite(fps)) {
                                        try {
                                            onVideoFps(Math.max(0, fps));
                                        } catch (err) {
                                            log('onVideoFps handler failed', err);
                                        }
                                    }
                                }
                            } else if (debug) {
                                log('Inbound video stats: none yet');
                            }
                            if (pair) {
                                log('Candidate pair:', {
                                    currentRoundTripTime: pair.currentRoundTripTime,
                                    totalRoundTripTime: pair.totalRoundTripTime,
                                    availableOutgoingBitrate: pair.availableOutgoingBitrate,
                                    availableIncomingBitrate: pair.availableIncomingBitrate,
                                });
                            }
                            if (debug) {
                                const v = videoRef.current;
                                if (v) {
                                    log('Video element:', {
                                        readyState: v.readyState,
                                        paused: v.paused,
                                        currentTime: v.currentTime,
                                        width: v.videoWidth,
                                        height: v.videoHeight,
                                    });
                                }
                            }
                        })
                        .catch((err) => log('getStats failed', err));
                }, interval);
            }
        };

        pc.ontrack = (event) => {
            log('Received track:', event.track.kind);
            const videoEl = videoRef.current;
            if (!videoEl) return;

            // Some senders (incl. GStreamer webrtcbin) may not populate event.streams.
            // Ensure we always attach a MediaStream so the <video> can render.
            const incoming = event.streams?.[0];
            const stream = incoming ?? remoteStreamRef.current ?? new MediaStream();
            remoteStreamRef.current = stream;

            if (!incoming) {
                try {
                    const existing = stream.getTracks();
                    if (!existing.some((t) => t.id === event.track.id)) {
                        stream.addTrack(event.track);
                    }
                } catch (err) {
                    log('Failed to add track to MediaStream', err);
                }
            }

            if (videoEl.srcObject !== stream) {
                videoEl.srcObject = stream;
                log('Attached stream to video element');
            }

            // Helpful event diagnostics.
            if (debug) {
                const onMeta = () => log('video loadedmetadata', { w: videoEl.videoWidth, h: videoEl.videoHeight, readyState: videoEl.readyState });
                const onResize = () => log('video resize', { w: videoEl.videoWidth, h: videoEl.videoHeight });
                const onPlaying = () => log('video playing');
                const onStalled = () => log('video stalled');
                const onWaiting = () => log('video waiting');
                const onError = () => log('video error', videoEl.error);
                videoEl.addEventListener('loadedmetadata', onMeta, { once: true });
                videoEl.addEventListener('resize', onResize);
                videoEl.addEventListener('playing', onPlaying);
                videoEl.addEventListener('stalled', onStalled);
                videoEl.addEventListener('waiting', onWaiting);
                videoEl.addEventListener('error', onError);
                // cleanup happens when component unmounts (video element replaced) or disconnect() nulls srcObject
            }

            // Kick autoplay; browsers can be finicky even with muted+playsInline.
            try {
                const p = videoEl.play();
                if (p && typeof p.then === 'function') {
                    p.catch((err) => log('video.play() failed', err));
                }
            } catch (err) {
                log('video.play() threw', err);
            }
        };

        return pc;
    }, [stunServers, signaling, log]);

    const connect = useCallback(async (): Promise<void> => {
        if (peerConnectionRef.current) {
            log('Already connected, disconnecting first');
            peerConnectionRef.current.close();
        }

        log('Creating new peer connection');
        const pc = createPeerConnection();
        peerConnectionRef.current = pc;

        // Add receive-only transceiver for video
        const tx = pc.addTransceiver('video', { direction: 'recvonly' });

        // Prefer H264 so the server answers with the matching codec
        try {
            const caps = RTCRtpReceiver.getCapabilities('video');
            const h264 = caps?.codecs?.filter(
                (c) =>
                    c.mimeType.toLowerCase() === 'video/h264' &&
                    (!c.sdpFmtpLine || c.sdpFmtpLine.includes('packetization-mode=1'))
            );
            if (h264?.length) {
                const others = caps!.codecs.filter((c) => !h264.includes(c));
                tx.setCodecPreferences([...h264, ...others]);
                log('Applied H264-first codec preferences');
            } else {
                log('No H264 codec found in capabilities; using defaults');
            }
        } catch (err) {
            log('Failed to set codec preferences', err);
        }

        try {
            log('Creating offer');
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            if (signaling && offer.sdp) {
                log('Sending offer to server');
                signaling.sendOffer(offer.sdp);
            }
        } catch (err) {
            console.error('[WebRTC] Failed to create offer:', err);
            throw err;
        }
    }, [createPeerConnection, signaling, log]);

    const disconnect = useCallback((): void => {
        log('Disconnecting');
        if (statsTimerRef.current) {
            window.clearInterval(statsTimerRef.current);
            statsTimerRef.current = null;
        }
        if (peerConnectionRef.current) {
            peerConnectionRef.current.close();
            peerConnectionRef.current = null;
        }
        lastVideoStatRef.current = {};
        remoteStreamRef.current = null;
        setConnectionState('new');
        setIsConnected(false);
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
    }, [log]);

    const handleAnswer = useCallback(async (sdp: string): Promise<void> => {
        const pc = peerConnectionRef.current;
        if (!pc) {
            console.warn('[WebRTC] No peer connection for answer');
            return;
        }

        try {
            log('Setting remote description (answer)');
            await pc.setRemoteDescription({
                type: 'answer',
                sdp,
            });
        } catch (err) {
            console.error('[WebRTC] Failed to set remote description:', err);
            throw err;
        }
    }, [log]);

    const handleIceCandidate = useCallback(async (candidate: RTCIceCandidateInit): Promise<void> => {
        const pc = peerConnectionRef.current;
        if (!pc) {
            console.warn('[WebRTC] No peer connection for ICE candidate');
            return;
        }

        try {
            log('Adding ICE candidate');
            await pc.addIceCandidate(candidate);
        } catch (err) {
            console.error('[WebRTC] Failed to add ICE candidate:', err);
            throw err;
        }
    }, [log]);

    return {
        videoRef,
        connectionState,
        isConnected,
        connect,
        disconnect,
        handleAnswer,
        handleIceCandidate,
    };
}

export default useWebRTCClient;
