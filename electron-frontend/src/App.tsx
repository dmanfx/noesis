import React, { useEffect, useState, useRef } from 'react';
import './style.css';
import { TelemetryProvider, useTelemetry } from './telemetry/TelemetryContext';
import { TelemetryDrawer } from './telemetry/TelemetryDrawer';
import { TelemetryToggle } from './telemetry/TelemetryToggle';

const WS_URL = 'ws://localhost:6008';

const Dashboard: React.FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const [systemStatus, setSystemStatus] = useState('Uptime: 0s');
  const [occupancy, setOccupancy] = useState('Loading...');
  const [activeTracks, setActiveTracks] = useState('Loading...');
  const [transitions, setTransitions] = useState('<li>Loading...</li>');
  const [kitchenFPS, setKitchenFPS] = useState('FPS: 0.0');
  const [livingRoomFPS, setLivingRoomFPS] = useState('FPS: 0.0');
  
  const socketRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const maxRetries = 10;
  const retryInterval = 5000;
  const { publish } = useTelemetry();

  const lastFpsUpdateTime = useRef(0);
  const FPS_UPDATE_INTERVAL = 2000; // 2 seconds

  // FPS tracking
  const fpsTrackingRef = useRef({
    'living-room': { frameCount: 0, lastUpdate: Date.now(), fps: 0, frameHistory: [] as number[] },
    'kitchen': { frameCount: 0, lastUpdate: Date.now(), fps: 0, frameHistory: [] as number[] }
  });

  const updateFPS = (cameraType: 'living-room' | 'kitchen') => {
    const tracker = fpsTrackingRef.current[cameraType];
    
    tracker.frameCount++;
    tracker.frameHistory.push(Date.now());
    
    const maxFrames = 30;
    if (tracker.frameHistory.length > maxFrames) {
      tracker.frameHistory.shift();
    }
    
    const now = Date.now();
    if (now - lastFpsUpdateTime.current < FPS_UPDATE_INTERVAL) {
        return;
    }
    lastFpsUpdateTime.current = now;

    if (now - tracker.lastUpdate >= 1000 || tracker.frameHistory.length >= maxFrames) {
      if (tracker.frameHistory.length >= 2) {
        const oldestFrame = tracker.frameHistory[0];
        const newestFrame = tracker.frameHistory[tracker.frameHistory.length - 1];
        const timeSpan = (newestFrame - oldestFrame) / 1000;
        
        if (timeSpan > 0) {
          tracker.fps = (tracker.frameHistory.length - 1) / timeSpan;
        }
      }
      
      if (cameraType === 'living-room') {
        setLivingRoomFPS(`FPS: ${tracker.fps.toFixed(1)}`);
        if (publish) {
          publish({ group: 'Camera Living Room', key: 'FPS', value: Number(tracker.fps.toFixed(1)), ts: Date.now() });
        }
      } else {
        setKitchenFPS(`FPS: ${tracker.fps.toFixed(1)}`);
        if (publish) {
          publish({ group: 'Camera Kitchen', key: 'FPS', value: Number(tracker.fps.toFixed(1)), ts: Date.now() });
        }
      }
      
      tracker.lastUpdate = now;
    }
  };

  const resetFPSTracking = () => {
    for (const cameraType in fpsTrackingRef.current) {
      fpsTrackingRef.current[cameraType as 'living-room' | 'kitchen'] = {
        frameCount: 0,
        lastUpdate: Date.now(),
        fps: 0,
        frameHistory: []
      };
    }
    setKitchenFPS('FPS: 0.0');
    setLivingRoomFPS('FPS: 0.0');
  };

  const processBinaryFrame = async (blob: Blob) => {
    try {
      // console.log('[WebSocket] Processing binary frame, blob size:', blob.size);
      const arrayBuffer = await blob.arrayBuffer();
      const dataView = new DataView(arrayBuffer);

      if (arrayBuffer.byteLength < 1) {
        console.error('[WebSocket] Received empty binary message.');
        return;
      }

      const cameraIdLen = dataView.getUint8(0);
      // console.log('[WebSocket] Camera ID length:', cameraIdLen);

      if (arrayBuffer.byteLength < 1 + cameraIdLen) {
        console.error(`[WebSocket] Binary message too short for camera ID. Length: ${arrayBuffer.byteLength}, ID length: ${cameraIdLen}`);
        return;
      }

      const cameraIdBytes = new Uint8Array(arrayBuffer, 1, cameraIdLen);
      const cameraId = new TextDecoder('utf-8').decode(cameraIdBytes);
      // console.log('[WebSocket] Camera ID:', cameraId);

      const jpegDataOffset = 1 + cameraIdLen;
      const jpegData = arrayBuffer.slice(jpegDataOffset);
      // console.log('[WebSocket] JPEG data size:', jpegData.byteLength);

      if (jpegData.byteLength === 0) {
        console.warn(`[WebSocket] Received binary message for camera '${cameraId}' with no JPEG data.`);
        return;
      }

      const jpegBlob = new Blob([jpegData], { type: 'image/jpeg' });
      displayFrame(cameraId, jpegBlob);

    } catch (error) {
      console.error('[WebSocket] Error processing binary frame:', error);
    }
  };

  const displayFrame = (cameraId: string, imageBlob: Blob) => {
    // console.log('[displayFrame] Called with camera:', cameraId, 'blob size:', imageBlob?.size);
    
    if (!imageBlob || !(imageBlob instanceof Blob)) {
      console.error(`Invalid image data for ${cameraId}, not a Blob:`, imageBlob);
      return;
    }

    try {
      const imageUrl = URL.createObjectURL(imageBlob);
      // console.log('[displayFrame] Created object URL:', imageUrl);
      
      const normalizedCamId = cameraId.toLowerCase().trim();
      // console.log('[displayFrame] Normalized camera ID:', normalizedCamId);
      
      let cameraType: 'living-room' | 'kitchen' | null = null;
      
      if (normalizedCamId === "rtsp_0" || normalizedCamId.includes("living") || normalizedCamId.includes("room1") || normalizedCamId === "1") {
        cameraType = 'living-room';
        // console.log('[displayFrame] Mapped to living room stream');
      } else if (normalizedCamId === "rtsp_1" || normalizedCamId.includes("kitchen") || normalizedCamId.includes("room2") || normalizedCamId === "2") {
        cameraType = 'kitchen';
        // console.log('[displayFrame] Mapped to kitchen stream');
      } else {
        console.warn(`Unknown camera ID: ${cameraId}`);
        URL.revokeObjectURL(imageUrl);
        return;
      }

      // Update the image source
      const imgElement = document.getElementById(cameraType === 'living-room' ? 'living-room-stream' : 'kitchen-stream') as HTMLImageElement;
      if (imgElement) {
        // Revoke previous URL if it exists
        if (imgElement.dataset.objectUrl) {
          URL.revokeObjectURL(imgElement.dataset.objectUrl);
        }
        imgElement.dataset.objectUrl = imageUrl;
        imgElement.src = imageUrl;
        
        // Update FPS tracking
        if (cameraType) {
          updateFPS(cameraType);
        }
      }

    } catch (error) {
      console.error(`Error in displayFrame for ${cameraId}:`, error);
    }
  };

  const handleStatsUpdate = (payload: any) => {
    if (!payload) return;

    // System Status & Uptime
    if (payload.uptime) {
      const uptimeSeconds = payload.uptime ?? 0;
      const hours = Math.floor(uptimeSeconds / 3600);
      const minutes = Math.floor((uptimeSeconds % 3600) / 60);
      const seconds = Math.floor(uptimeSeconds % 60);
      setSystemStatus(`Uptime: ${hours}h ${minutes}m ${seconds}s`);
      
      // Publish to telemetry
      if (publish) {
        publish({ group: 'Application', key: 'Uptime', value: uptimeSeconds, ts: Date.now() });
      }
    }

    // Application-level stats
    if (payload.application) {
      if (publish) {
        publish({ group: 'Application', key: 'Running', value: payload.application.running ? 'Yes' : 'No', ts: Date.now() });
        publish({ group: 'Application', key: 'Active Cameras', value: payload.application.cameras_active, ts: Date.now() });
        publish({ group: 'Application', key: 'Active Processors', value: payload.application.processors_active, ts: Date.now() });
      }
    }

    // Process camera data
    if (payload.cameras && typeof payload.cameras === 'object') {
      let globalOccupancy: { [key: string]: number } = {}; 
      let allActiveTracks: any[] = [];
      let allTransitions: any[] = [];
      
      const now = Date.now();
      const shouldUpdateFps = now - lastFpsUpdateTime.current > FPS_UPDATE_INTERVAL;

      for (const cameraId in payload.cameras) {
        const cameraData = payload.cameras[cameraId];
        if (!cameraData) continue;

        // Throttled FPS state updates
        if (shouldUpdateFps) {
          if (cameraId === 'kitchen') {
            setKitchenFPS(`FPS: ${cameraData.fps?.toFixed(1) ?? '0.0'}`);
          } else if (cameraId === 'living-room') {
            setLivingRoomFPS(`FPS: ${cameraData.fps?.toFixed(1) ?? '0.0'}`);
          }
        }

        // Publish basic camera stats (always publish for telemetry)
        const groupName = `Camera ${cameraId}`;
        if (publish) {
          if (typeof cameraData.frame_count !== 'undefined') {
            publish({ group: groupName, key: 'Frames', value: cameraData.frame_count, ts: Date.now() });
          }
          if (typeof cameraData.processing_time_ms !== 'undefined') {
            publish({ group: groupName, key: 'Proc ms', value: cameraData.processing_time_ms, ts: Date.now() });
          }
          if (typeof cameraData.fps !== 'undefined') {
            publish({ group: groupName, key: 'FPS', value: Number(cameraData.fps.toFixed(1)), ts: Date.now() });
          }
          if (typeof cameraData.status !== 'undefined') {
            publish({ group: groupName, key: 'Status', value: cameraData.status, ts: Date.now() });
          }
        }

        // Process tracking data
        const trackingData = cameraData.tracking;
        if (trackingData && typeof trackingData === 'object') {
          if (trackingData.occupancy && typeof trackingData.occupancy === 'object') {
            for (const zone in trackingData.occupancy) {
              globalOccupancy[zone] = (globalOccupancy[zone] || 0) + trackingData.occupancy[zone];
            }
          }
          
          if (trackingData.active_tracks && Array.isArray(trackingData.active_tracks)) {
            allActiveTracks = allActiveTracks.concat(trackingData.active_tracks);
          }

          if (trackingData.transitions && Array.isArray(trackingData.transitions)) {
            allTransitions = allTransitions.concat(trackingData.transitions);
          }
        }
      }

      if (shouldUpdateFps) {
        lastFpsUpdateTime.current = now;
      }

      // Update Occupancy Display
      let occupancyHTML = '<ul style="margin: 0; padding-left: 15px;">';
      const sortedZones = Object.entries(globalOccupancy).sort(([, a], [, b]) => b - a);
      if (sortedZones.length > 0) {
        sortedZones.forEach(([zone, count]) => {
          occupancyHTML += `<li><strong>${zone}:</strong> <span style="display: inline-block; min-width: 3ch; text-align: right;">${count}</span></li>`;
          if (publish) {
            publish({ group: 'Occupancy', key: zone, value: count, ts: Date.now() });
          }
        });
      } else {
        occupancyHTML += '<li>No occupancy data.</li>';
        if (publish) {
          publish({ group: 'Occupancy', key: 'none', value: 0, ts: Date.now() });
        }
      }
      occupancyHTML += '</ul>';
      setOccupancy(occupancyHTML);

      // Update Active Tracks Display
      let tracksHTML = '';
      if (allActiveTracks.length > 0) {
        allActiveTracks.sort((a, b) => (a.track_id || 0) - (b.track_id || 0)).forEach(track => {
          const dwellText = track.dwell_time?.toFixed(1) ?? '0.0';
          const center = track.center || ['N/A', 'N/A'];
          const velocity = track.velocity || [0, 0];
          const speed = Math.sqrt(velocity[0]**2 + velocity[1]**2);
          const speedText = speed?.toFixed(1) ?? '0.0';
          
          tracksHTML += `<div class="track-detail-item">`;
          tracksHTML += `<strong>ID ${track.track_id || 'N/A'} (${track.camera_id || 'N/A'}):</strong><br>`;
          tracksHTML += `Zone: ${track.zone || '-'}, Dwell: <span style="display: inline-block; min-width: 4ch; text-align: right;">${dwellText}</span>s<br>`;
          tracksHTML += `Pos: [${center[0]}, ${center[1]}], Speed: <span style="display: inline-block; min-width: 4ch; text-align: right;">${speedText}</span> px/s`;
          tracksHTML += `</div>`;
        });
      } else {
        tracksHTML = '<span>No active tracks.</span>';
      }
      setActiveTracks(tracksHTML);

      // Publish tracking telemetry
      if (publish) {
        publish({ group: 'Tracking', key: 'Active Tracks', value: allActiveTracks.length, ts: Date.now() });
        publish({ group: 'Tracking', key: 'Transitions', value: allTransitions.length, ts: Date.now() });
      }

      // Update Transitions Display
      let transitionsHTML = '';
      if (allTransitions.length > 0) {
        allTransitions.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
          .slice(0, 10)
          .forEach(t => {
            const timeStr = t.timestamp ? new Date(t.timestamp * 1000).toLocaleTimeString() : '??:??:??';
            transitionsHTML += `<li>[${timeStr}] ID ${t.track_id || 'N/A'} (${t.camera_id || 'N/A'}): ${t.from_zone || 'N/A'} → ${t.to_zone || 'N/A'}</li>`;
          });
      } else {
        transitionsHTML = '<li>No recent transitions.</li>';
      }
      setTransitions(transitionsHTML);
    } else {
      // Handle case where payload.cameras is missing or not an object
      console.warn("Stats payload missing or invalid 'cameras' structure.");
      // Reset telemetry for missing data
      if (publish) {
        publish({ group: 'Tracking', key: 'Active Tracks', value: 0, ts: Date.now() });
        publish({ group: 'Tracking', key: 'Transitions', value: 0, ts: Date.now() });
      }
    }

    // Publish connection status
    if (publish) {
      publish({ group: 'Connection', key: 'Status', value: 'Connected', ts: Date.now() });
    }
  };

  const connectWebSocket = () => {
    console.log(`Attempting to connect to ${WS_URL}...`);
    setConnectionStatus(`Connecting to ${WS_URL}...`);

    socketRef.current = new WebSocket(WS_URL);

    socketRef.current.onopen = () => {
      console.log('WebSocket connection opened');
      setConnectionStatus('Connected');
      retryCountRef.current = 0;
      resetFPSTracking();
      
      // Publish connection status to telemetry
      if (publish) {
        publish({ group: 'Connection', key: 'Status', value: 'Connected', ts: Date.now() });
      }
    };

    socketRef.current.onclose = (event: CloseEvent) => {
      console.log('WebSocket connection closed:', event.code, event.reason);
      setConnectionStatus(`Disconnected (Code: ${event.code}). Retrying...`);
      
      // Publish connection status to telemetry
      if (publish) {
        publish({ group: 'Connection', key: 'Status', value: 'Disconnected', ts: Date.now() });
      }
      
      // Clear images on disconnect
      const livingRoomImg = document.getElementById('living-room-stream') as HTMLImageElement;
      const kitchenImg = document.getElementById('kitchen-stream') as HTMLImageElement;
      if (livingRoomImg) livingRoomImg.src = "";
      if (kitchenImg) kitchenImg.src = "";
      
      resetFPSTracking();
      
      // Attempt to reconnect
      if (retryCountRef.current < maxRetries) {
        retryCountRef.current++;
        console.log(`Attempting reconnect #${retryCountRef.current} in ${retryInterval / 1000} seconds...`);
        setTimeout(connectWebSocket, retryInterval);
      } else {
        console.error(`Max retries (${maxRetries}) reached. Stopping reconnection attempts.`);
        setConnectionStatus('Disconnected (Max retries reached)');
      }
    };

    socketRef.current.onerror = (error: Event) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('Error (Check console)');
      
      // Publish connection error to telemetry
      if (publish) {
        publish({ group: 'Connection', key: 'Status', value: 'Error', ts: Date.now() });
      }
    };

    socketRef.current.onmessage = (event: MessageEvent) => {
      if (event.data instanceof Blob) {
        processBinaryFrame(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], { type: 'application/octet-stream' });
        processBinaryFrame(blob);
      } else {
        try {
          const data = JSON.parse(event.data);
          // console.log('[WebSocket] Parsed JSON message:', data);
          
          if (data.type === 'stats' && data.payload) {
            handleStatsUpdate(data.payload);
          } else if (data.type === 'frame') {
            // console.log('[WebSocket] Received JSON frame format:', data);
            // Handle JSON frame format if needed
          } else if (data.type === 'detection_config_sync') {
            // console.log('[WebSocket] Received detection config sync:', data.config);
            // Handle detection config sync
          } else if (data.type === 'detection_config_update') {
            // console.log('[WebSocket] Received detection config update:', data.config);
            // Handle detection config update
          } else if (data.type === 'detection_toggle_update') {
            // console.log('[WebSocket] Received detection toggle update:', data);
            // Handle detection toggle update
          } else {
            // console.log('[WebSocket] Received unknown JSON message format:', data);
          }
        } catch (e) {
          console.error('[WebSocket] JSON parse error:', e, event.data);
        }
      }
    };
  };

  const clearStats = () => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log('Sending clear_stats message to backend...');
      socketRef.current.send(JSON.stringify({ type: 'clear_stats' }));
    } else {
      console.warn('WebSocket not connected. Cannot send clear_stats message.');
    }
  };

  useEffect(() => {
    // Start WebSocket connection
    connectWebSocket();

    // Cleanup on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  // Clock update effect
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      const seconds = String(now.getSeconds()).padStart(2, '0');
      const clockElement = document.getElementById('clock');
      if (clockElement) {
        clockElement.textContent = `${hours}:${minutes}:${seconds}`;
      }
    };

    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  // Fullscreen controls effect
  useEffect(() => {
    const fullscreenBtns = document.querySelectorAll('.fullscreen-btn');
    
    const handleFullscreen = (button: Element) => {
      const targetId = button.getAttribute('data-target');
      const targetElement = document.getElementById(targetId!);
      
      if (targetElement) {
        const isMaximized = targetElement.classList.contains('maximized-stream');
        
        let perfStatsElem = null;
        if (targetId === 'kitchen-stream') {
          perfStatsElem = document.getElementById('kitchen-perf');
        } else if (targetId === 'living-room-stream') {
          perfStatsElem = document.getElementById('living-room-perf');
        }
        
        // Reset all videos to normal state first
        document.querySelectorAll('.maximized-stream').forEach(stream => {
          stream.classList.remove('maximized-stream');
        });
        
        // Reset all buttons to normal state
        document.querySelectorAll('.fullscreen-btn-maximized').forEach(btn => {
          btn.classList.remove('fullscreen-btn-maximized');
        });
        
        // Reset all performance stats to normal state
        document.querySelectorAll('.perf-stats-maximized').forEach(stats => {
          stats.classList.remove('perf-stats-maximized');
        });
        
        // Toggle maximized state for this video
        if (!isMaximized) {
          targetElement.classList.add('maximized-stream');
          button.classList.add('fullscreen-btn-maximized');
          if (perfStatsElem) {
            perfStatsElem.classList.add('perf-stats-maximized');
          }
        }
      }
    };

    fullscreenBtns.forEach(button => {
      button.addEventListener('click', () => handleFullscreen(button));
    });

    // Keyboard event handler
    const handleKeydown = (event: KeyboardEvent) => {
      // Handle Escape key for fullscreen
      if (event.key === 'Escape') {
        document.querySelectorAll('.maximized-stream').forEach(stream => {
          stream.classList.remove('maximized-stream');
        });
        
        document.querySelectorAll('.fullscreen-btn-maximized').forEach(btn => {
          btn.classList.remove('fullscreen-btn-maximized');
        });
        
        document.querySelectorAll('.perf-stats-maximized').forEach(stats => {
          stats.classList.remove('perf-stats-maximized');
        });
      }
      
      // Handle 'T' key for telemetry drawer toggle
      if (event.key.toLowerCase() === 't') {
        // Only trigger if not typing in an input field
        const activeElement = document.activeElement;
        const isInputField = activeElement && (
          activeElement.tagName === 'INPUT' || 
          activeElement.tagName === 'TEXTAREA' || 
          (activeElement as HTMLElement).contentEditable === 'true'
        );
        
        if (!isInputField) {
          event.preventDefault();
          setDrawerOpen((prev: boolean) => !prev);
        }
      }
    };

    document.addEventListener('keydown', handleKeydown);

    return () => {
      document.removeEventListener('keydown', handleKeydown);
    };
  }, []);

  // Detection controls effect
  useEffect(() => {
    const sendDetectionToggle = (toggleName: string, enabled: boolean) => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const message = {
          type: 'set_detection_toggle',
          toggle_name: toggleName,
          enabled: enabled
        };
        console.log('Sending detection toggle message:', message);
        socketRef.current.send(JSON.stringify(message));
      }
    };

    const sendDetectionConfig = (config: any) => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const message = {
          type: 'update_detection_config',
          config: config
        };
        console.log('Sending detection config message:', message);
        socketRef.current.send(JSON.stringify(message));
      }
    };

    // Detection type toggles
    const toggleDetectPeople = document.getElementById('toggle-detect-people') as HTMLInputElement;
    if (toggleDetectPeople) {
      toggleDetectPeople.addEventListener('change', (event) => {
        sendDetectionToggle('detect_people', (event.target as HTMLInputElement).checked);
      });
    }
    
    const toggleDetectVehicles = document.getElementById('toggle-detect-vehicles') as HTMLInputElement;
    if (toggleDetectVehicles) {
      toggleDetectVehicles.addEventListener('change', (event) => {
        sendDetectionToggle('detect_vehicles', (event.target as HTMLInputElement).checked);
      });
    }
    
    const toggleDetectFurniture = document.getElementById('toggle-detect-furniture') as HTMLInputElement;
    if (toggleDetectFurniture) {
      toggleDetectFurniture.addEventListener('change', (event) => {
        sendDetectionToggle('detect_furniture', (event.target as HTMLInputElement).checked);
      });
    }
    
    // Detection settings sliders
    const confidenceThreshold = document.getElementById('confidence-threshold') as HTMLInputElement;
    const confidenceValue = document.getElementById('confidence-value');
    if (confidenceThreshold && confidenceValue) {
      confidenceThreshold.addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        confidenceValue!.textContent = value.toFixed(2);
        sendDetectionConfig({ confidence_threshold: value });
      });
    }
    
    const iouThreshold = document.getElementById('iou-threshold') as HTMLInputElement;
    const iouValue = document.getElementById('iou-value');
    if (iouThreshold && iouValue) {
      iouThreshold.addEventListener('input', (event) => {
        const value = parseFloat((event.target as HTMLInputElement).value);
        iouValue!.textContent = value.toFixed(2);
        sendDetectionConfig({ iou_threshold: value });
      });
    }
    
    // Detection enable/disable toggle
    const toggleDetectionEnabled = document.getElementById('toggle-detection-enabled') as HTMLInputElement;
    if (toggleDetectionEnabled) {
      toggleDetectionEnabled.addEventListener('change', (event) => {
        sendDetectionConfig({ detection_enabled: (event.target as HTMLInputElement).checked });
      });
    }
  }, []);

  // Collapsible controls effect
  useEffect(() => {
    const controlsHeader = document.getElementById('controls-header');
    const controlsContent = document.getElementById('controls-content');
    const controlsSection = document.querySelector('.controls-section');
    const collapseToggle = document.getElementById('controls-collapse-toggle');
    
    if (controlsHeader && controlsContent && controlsSection && collapseToggle) {
      let isCollapsed = true;
      
      const toggleCollapse = () => {
        isCollapsed = !isCollapsed;
        
        if (isCollapsed) {
          controlsContent.classList.remove('expanded');
          controlsSection.classList.add('collapsed');
          collapseToggle.textContent = '+';
        } else {
          controlsContent.classList.add('expanded');
          controlsSection.classList.remove('collapsed');
          collapseToggle.textContent = '−';
        }
      };
      
      // Apply initial collapsed state
      controlsSection.classList.add('collapsed');
      collapseToggle.textContent = '+';
      
      // Add click event listeners
      controlsHeader.addEventListener('click', (event) => {
        if (event.target !== collapseToggle && !collapseToggle.contains(event.target as Node)) {
          toggleCollapse();
        }
      });
      
      collapseToggle.addEventListener('click', (event) => {
        event.stopPropagation();
        toggleCollapse();
      });
    }
  }, []);

  return (
    <div id="app-container">
      <header>
        <h1>Smart Room Dashboard</h1>
        <div id="system-status">{systemStatus}</div>
        <div id="clock">--:--:--</div>
      </header>

      <main>
        <div className="video-section">
          <div className="video-container">
            <h2>Kitchen</h2>
            <div className="video-wrapper">
              <img id="kitchen-stream" src="" alt="Kitchen Stream" width="640" height="360" />
              <button className="fullscreen-btn" data-target="kitchen-stream">Max</button>
            </div>
            <div id="kitchen-perf" className="perf-stats" style={{ minWidth: '80px', textAlign: 'right' }}>{kitchenFPS}</div>
          </div>

          <div className="video-container">
            <h2>Living Room</h2>
            <div className="video-wrapper">
              <img id="living-room-stream" src="" alt="Living Room Stream" width="640" height="360" />
              <button className="fullscreen-btn" data-target="living-room-stream">Max</button>
            </div>
            <div id="living-room-perf" className="perf-stats" style={{ minWidth: '80px', textAlign: 'right' }}>{livingRoomFPS}</div>
          </div>
        </div>

        <div className="stats-section">
          <div className="stats-card" id="occupancy-card">
            <h3>Zone Occupancy</h3>
            <div id="occupancy-content" dangerouslySetInnerHTML={{ __html: occupancy }}></div>
          </div>
          <div className="stats-card" id="track-details-card">
            <h3>Active Tracks</h3>
            <div id="track-details-content" dangerouslySetInnerHTML={{ __html: activeTracks }}></div>
          </div>
          <div className="stats-card" id="transitions-card">
            <h3>Recent Transitions</h3>
            <ul id="transitions-list" dangerouslySetInnerHTML={{ __html: transitions }}></ul>
          </div>
        </div>

        <div className="controls-section" id="controls-section">
          <div className="controls-header" id="controls-header">
            <button className="collapse-toggle" id="controls-collapse-toggle">−</button>
            <h3>Detection Controls</h3>
          </div>
          <div className="controls-content" id="controls-content">
            <div className="toggle-group">
              <h4>Detection Types</h4>
              <label className="toggle-label">
                <input type="checkbox" id="toggle-detect-people" defaultChecked />
                <span className="toggle-slider"></span>
                Detect People
              </label>
              <label className="toggle-label">
                <input type="checkbox" id="toggle-detect-vehicles" />
                <span className="toggle-slider"></span>
                Detect Vehicles
              </label>
              <label className="toggle-label">
                <input type="checkbox" id="toggle-detect-furniture" />
                <span className="toggle-slider"></span>
                Detect Furniture
              </label>
            </div>

            <div className="settings-group">
              <h4>Detection Settings</h4>
              <div className="setting-item">
                <label htmlFor="confidence-threshold">Confidence Threshold:</label>
                <input type="range" id="confidence-threshold" min="0.1" max="0.9" step="0.05" defaultValue="0.3" />
                <span id="confidence-value">0.3</span>
              </div>
              <div className="setting-item">
                <label htmlFor="iou-threshold">IOU Threshold:</label>
                <input type="range" id="iou-threshold" min="0.1" max="0.9" step="0.05" defaultValue="0.45" />
                <span id="iou-value">0.45</span>
              </div>
              <div className="setting-item">
                <label className="toggle-label">
                  <input type="checkbox" id="toggle-detection-enabled" defaultChecked />
                  <span className="toggle-slider"></span>
                  Enable Detection
                </label>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer>
        <span id="connection-status">{connectionStatus}</span>
        <button id="clear-stats-btn" className="footer-button" onClick={clearStats}>Clear Stats</button>
      </footer>
      <TelemetryToggle onToggle={() => setDrawerOpen(o => !o)} />
      <TelemetryDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </div>
  );
};

const App: React.FC = () => (
  <TelemetryProvider>
    <Dashboard />
  </TelemetryProvider>
);

export default App;

