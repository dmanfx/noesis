// renderer.js - Electron Renderer Process Logic

const WS_URL = 'ws://localhost:6008'; // Default, adjust if backend is elsewhere
let socket = null;
// Remove redundant/unused variables for last URLs
// let lastLivingRoomUrl = null;
// let lastKitchenUrl = null;
let retryInterval = 5000; // 5 seconds
let maxRetries = 10;
let currentRetries = 0;
let connectTimeout;

// --- DOM Elements --- CORRECTED IDs ---
const connectionStatusElem = document.getElementById('connection-status');
// Use the correct IDs from index.html
const livingRoomStreamElem = document.getElementById('living-room-stream');
const kitchenStreamElem = document.getElementById('kitchen-stream');
// const livingRoomOverlayElem = document.getElementById('living-room-overlay'); // REMOVED
// const kitchenOverlayElem = document.getElementById('kitchen-overlay'); // REMOVED
const livingRoomPerfElem = document.getElementById('living-room-perf');
const kitchenPerfElem = document.getElementById('kitchen-perf');
const occupancyContentElem = document.getElementById('occupancy-content');
const trackDetailsContentElem = document.getElementById('track-details-content');
const transitionsListElem = document.getElementById('transitions-list');
const systemStatusElem = document.getElementById('system-status');
const clockElem = document.getElementById('clock');
const fullscreenBtns = document.querySelectorAll('.fullscreen-btn');
const clearStatsButton = document.getElementById('clear-stats-btn');

// --- FPS Tracking Variables ---
const fpsTracking = {
    'living-room': {
        frameCount: 0,
        lastUpdate: Date.now(),
        fps: 0,
        frameHistory: []
    },
    'kitchen': {
        frameCount: 0,
        lastUpdate: Date.now(),
        fps: 0,
        frameHistory: []
    }
};

// --- Canvas Contexts (Ensure overlay elements exist first) --- // REMOVED
// const livingRoomCtx = livingRoomOverlayElem ? livingRoomOverlayElem.getContext('2d') : null; // REMOVED
// const kitchenCtx = kitchenOverlayElem ? kitchenOverlayElem.getContext('2d') : null; // REMOVED

// --- FPS Calculation Functions ---
function updateFPS(cameraType) {
    const now = Date.now();
    const tracker = fpsTracking[cameraType];
    
    if (!tracker) return;
    
    // Increment frame count
    tracker.frameCount++;
    
    // Add current timestamp to frame history
    tracker.frameHistory.push(now);
    
    // Keep only last 30 frames for calculation (rolling window)
    const maxFrames = 30;
    if (tracker.frameHistory.length > maxFrames) {
        tracker.frameHistory.shift();
    }
    
    // Calculate FPS every second or when we have enough frames
    if (now - tracker.lastUpdate >= 1000 || tracker.frameHistory.length >= maxFrames) {
        if (tracker.frameHistory.length >= 2) {
            // Calculate FPS based on frame history
            const oldestFrame = tracker.frameHistory[0];
            const newestFrame = tracker.frameHistory[tracker.frameHistory.length - 1];
            const timeSpan = (newestFrame - oldestFrame) / 1000; // Convert to seconds
            
            if (timeSpan > 0) {
                tracker.fps = (tracker.frameHistory.length - 1) / timeSpan;
            }
        }
        
        // Update display
        updateFPSDisplay(cameraType, tracker.fps);
        tracker.lastUpdate = now;
    }
}

function updateFPSDisplay(cameraType, fps) {
    let perfElem = null;
    
    if (cameraType === 'living-room') {
        perfElem = livingRoomPerfElem;
    } else if (cameraType === 'kitchen') {
        perfElem = kitchenPerfElem;
    }
    
    if (perfElem) {
        perfElem.textContent = `FPS: ${fps.toFixed(1)}`;
    }
}

function resetFPSTracking() {
    for (const cameraType in fpsTracking) {
        fpsTracking[cameraType].frameCount = 0;
        fpsTracking[cameraType].lastUpdate = Date.now();
        fpsTracking[cameraType].fps = 0;
        fpsTracking[cameraType].frameHistory = [];
        updateFPSDisplay(cameraType, 0);
    }
}

// --- WebSocket Connection ---
function connectWebSocket() {
    console.log(`Attempting to connect to ${WS_URL}...`);
    // Use connectionStatusElem consistently
    if (connectionStatusElem) {
        connectionStatusElem.textContent = `Status: Connecting to ${WS_URL}...`;
        connectionStatusElem.style.color = 'orange';
    }

    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        console.log('WebSocket connection opened');
        if (connectionStatusElem) {
            connectionStatusElem.textContent = 'Status: Connected';
            connectionStatusElem.style.color = 'green';
        }
        currentRetries = 0; // Reset retries
        resetStatsDisplay(); // Reset stats on successful connect
        resetFPSTracking(); // Reset FPS tracking on successful connect
    };

    socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event.code, event.reason);
        if (connectionStatusElem) {
            connectionStatusElem.textContent = `Status: Disconnected (Code: ${event.code}). Retrying...`;
            connectionStatusElem.style.color = 'red';
        }
        if(livingRoomStreamElem) livingRoomStreamElem.src = ""; // Clear images on disconnect
        if(kitchenStreamElem) kitchenStreamElem.src = "";
        resetFPSTracking(); // Reset FPS tracking on disconnect
        // Attempt to reconnect
        if (currentRetries < maxRetries) {
            currentRetries++;
            console.log(`Attempting reconnect #${currentRetries} in ${retryInterval / 1000} seconds...`);
            setTimeout(connectWebSocket, retryInterval);
        } else {
            console.error(`Max retries (${maxRetries}) reached. Stopping reconnection attempts.`);
            if (connectionStatusElem) connectionStatusElem.textContent = `Status: Disconnected (Max retries reached)`;
        }
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (connectionStatusElem) {
            connectionStatusElem.textContent = 'Status: Error (Check console)';
            connectionStatusElem.style.color = 'red';
        }
        // onclose will likely follow
    };

    socket.onmessage = (event) => {
        // Check if it's binary data (JPEG frame) or text (JSON)
        if (event.data instanceof Blob) {
            // This is a binary frame - process it using existing function
            processBinaryFrame(event.data);
        } else if (event.data instanceof ArrayBuffer) {
            // Convert ArrayBuffer to Blob for processing
            const blob = new Blob([event.data], { type: 'application/octet-stream' });
            processBinaryFrame(blob);
        } else {
            // This is JSON data
            try {
                const data = JSON.parse(event.data);
                console.log('[WebSocket] Parsed JSON message:', data);
                
                if (data.type === 'stats' && data.payload) {
                    handleStatsUpdate(data.payload);
                } else if (data.type === 'frame') {
                    // Handle JSON frame format (fallback for backwards compatibility)
                    console.log('[WebSocket] Received JSON frame format:', data);
                    displayFrameFromJSON(data);
                } else if (data.type === 'detection_config_sync') {
                    // Handle initial detection configuration sync
                    console.log('[WebSocket] Received detection config sync:', data.config);
                    syncDetectionConfig(data.config);
                } else if (data.type === 'detection_config_update') {
                    // Handle detection configuration updates
                    console.log('[WebSocket] Received detection config update:', data.config);
                    syncDetectionConfig(data.config);
                } else if (data.type === 'detection_toggle_update') {
                    // Handle detection toggle updates
                    console.log('[WebSocket] Received detection toggle update:', data);
                    syncDetectionToggle(data.toggle_name, data.enabled);
                } else {
                    console.log('[WebSocket] Received unknown JSON message format:', data);
                }
            } catch (e) {
                console.error('[WebSocket] JSON parse error:', e, event.data);
            }
        }
    };
}

// --- ADDED: Function to process binary frame data ---
async function processBinaryFrame(blob) {
    try {
        console.log('[WebSocket] Processing binary frame, blob size:', blob.size);
        const arrayBuffer = await blob.arrayBuffer();
        const dataView = new DataView(arrayBuffer);

        if (arrayBuffer.byteLength < 1) {
            console.error('[WebSocket] Received empty binary message.');
            return;
        }

        // 1. Read camera ID length (first byte)
        const cameraIdLen = dataView.getUint8(0);
        console.log('[WebSocket] Camera ID length:', cameraIdLen);

        if (arrayBuffer.byteLength < 1 + cameraIdLen) {
            console.error(`[WebSocket] Binary message too short for camera ID. Length: ${arrayBuffer.byteLength}, ID length: ${cameraIdLen}`);
            return;
        }

        // 2. Read camera ID string
        const cameraIdBytes = new Uint8Array(arrayBuffer, 1, cameraIdLen);
        const cameraId = new TextDecoder('utf-8').decode(cameraIdBytes);
        console.log('[WebSocket] Camera ID:', cameraId);

        // 3. Extract JPEG data (the rest of the buffer)
        const jpegDataOffset = 1 + cameraIdLen;
        const jpegData = arrayBuffer.slice(jpegDataOffset);
        console.log('[WebSocket] JPEG data size:', jpegData.byteLength);

        if (jpegData.byteLength === 0) {
             console.warn(`[WebSocket] Received binary message for camera '${cameraId}' with no JPEG data.`);
             return;
        }

        // 4. Create a Blob for the JPEG data
        const jpegBlob = new Blob([jpegData], { type: 'image/jpeg' });

        // 5. Display the frame
        console.log('[WebSocket] Calling displayFrame with camera:', cameraId);
        displayFrame(cameraId, jpegBlob);

    } catch (error) {
        console.error('[WebSocket] Error processing binary frame:', error);
    }
}

// --- UI Update Functions ---

// --- ADDED: Function to display frame from JSON data ---
function displayFrameFromJSON(frameData) {
    if (!frameData || !frameData.camera_id || !frameData.image) {
        console.error('Invalid JSON frame data:', frameData);
        return;
    }
    
    try {
        // Convert base64 to blob
        const base64Data = frameData.image;
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const imageBlob = new Blob([bytes], { type: 'image/jpeg' });
        
        // Use existing display function
        displayFrame(frameData.camera_id, imageBlob);
        
    } catch (error) {
        console.error('Error processing JSON frame data:', error);
    }
}

// --- MODIFIED: displayFrame now accepts a Blob ---
function displayFrame(cameraId, imageBlob) { 
    console.log('[displayFrame] Called with camera:', cameraId, 'blob size:', imageBlob?.size);
    
    if (!imageBlob || !(imageBlob instanceof Blob)) {
        console.error(`Invalid image data for ${cameraId}, not a Blob:`, imageBlob);
        return;
    }

    try {
        // Create an Object URL for the Blob
        const imageUrl = URL.createObjectURL(imageBlob);
        console.log('[displayFrame] Created object URL:', imageUrl);
        
        // Get the correct image element and camera type for FPS tracking
        let imgElement = null;
        let cameraType = null;
        const normalizedCamId = cameraId.toLowerCase().trim();
        console.log('[displayFrame] Normalized camera ID:', normalizedCamId);
        
        // --- MODIFIED: Updated camera ID mapping ---
        if (normalizedCamId === "rtsp_0" || normalizedCamId.includes("living") || normalizedCamId.includes("room1") || normalizedCamId === "1") {
            imgElement = livingRoomStreamElem;
            cameraType = 'living-room';
            console.log('[displayFrame] Mapped to living room stream');
        } else if (normalizedCamId === "rtsp_1" || normalizedCamId.includes("kitchen") || normalizedCamId.includes("room2") || normalizedCamId === "2") {
            imgElement = kitchenStreamElem;
            cameraType = 'kitchen';
            console.log('[displayFrame] Mapped to kitchen stream');
        } else {
            console.warn(`Unknown camera ID: ${cameraId}`);
            URL.revokeObjectURL(imageUrl); // Revoke URL if not used
            return;
        }
        // --- END MODIFIED ---

        if (!imgElement) {
            console.error(`Image element for ${cameraId} is null/undefined!`);
             URL.revokeObjectURL(imageUrl); // Revoke URL if not used
            return;
        }
        
        console.log('[displayFrame] Found image element:', imgElement.id);

        // --- ADDED: Revoke previous URL if it exists to prevent leaks ---
        if (imgElement.dataset.objectUrl) {
            URL.revokeObjectURL(imgElement.dataset.objectUrl);
        }
        imgElement.dataset.objectUrl = imageUrl; // Store the new URL
        // --- END ADDED ---

        // Set the image source
        imgElement.src = imageUrl;
        
        // Update FPS tracking for this camera
        if (cameraType) {
            updateFPS(cameraType);
        }
        
        // --- ADDED: Revoke the Object URL once the image is loaded --- 
        imgElement.onload = () => {
            // No need to revoke here if we revoke the *previous* one before setting src
            // URL.revokeObjectURL(imageUrl); 
            // We can clear the dataset attribute after load if desired
            // delete imgElement.dataset.objectUrl;
        };
        imgElement.onerror = (err) => {
            console.error(`Error loading image for ${cameraId}:`, err);
            // URL.revokeObjectURL(imageUrl); // Revoke on error too
             delete imgElement.dataset.objectUrl; // Clean up dataset
        };
        // --- END ADDED ---

    } catch (error) {
        console.error(`Error in displayFrame for ${cameraId}:`, error);
        // Ensure potential object URLs are revoked if an error occurs mid-function
        if (typeof imageUrl !== 'undefined') {
            URL.revokeObjectURL(imageUrl);
        }
    }
}

// Use correct element variable names from top
function handleStatsUpdate(payload) {
    if (!payload) return;

    // --- DEBUG: Add explicit payload structure logging ---
    console.log("Stats payload structure:", {
        hasPayload: !!payload, 
        hasCameras: !!payload.cameras, 
        cameraIds: payload.cameras ? Object.keys(payload.cameras) : [],
        cameraDataSample: payload.cameras && Object.keys(payload.cameras).length > 0 
            ? {
                sampleCameraId: Object.keys(payload.cameras)[0],
                hasTracking: !!payload.cameras[Object.keys(payload.cameras)[0]].tracking,
                trackingKeys: payload.cameras[Object.keys(payload.cameras)[0]].tracking 
                    ? Object.keys(payload.cameras[Object.keys(payload.cameras)[0]].tracking) 
                    : "No tracking object",
                activeTracks: payload.cameras[Object.keys(payload.cameras)[0]].tracking?.active_tracks
                    ? payload.cameras[Object.keys(payload.cameras)[0]].tracking.active_tracks.length
                    : "No active_tracks array"
              } 
            : "No cameras available"
    });

    // System Status & Uptime (Assuming these are top-level in payload)
    if (payload.uptime && systemStatusElem) {
        const uptimeSeconds = payload.uptime ?? 0;
        const hours = Math.floor(uptimeSeconds / 3600);
        const minutes = Math.floor((uptimeSeconds % 3600) / 60);
        const seconds = Math.floor(uptimeSeconds % 60); // Use floor for integer seconds
        systemStatusElem.textContent = `Uptime: ${hours}h ${minutes}m ${seconds}s`;
    }

    // Performance, Occupancy, Active Tracks, Transitions (Now nested under payload.cameras[cameraId].tracking)
    if (payload.cameras && typeof payload.cameras === 'object') {
        // Initialize aggregation variables OUTSIDE the loop
        let globalOccupancy = {}; 
        let allActiveTracks = [];
        let allTransitions = [];
        
        // --- Process Camera Specific Data ---
        for (const cameraId in payload.cameras) {
            const cameraData = payload.cameras[cameraId];
            if (!cameraData) continue;

            // --- Performance is now handled by client-side FPS tracking ---
            // Backend FPS data is no longer used since we calculate it client-side
            // This provides more accurate frontend FPS measurements
            
            // --- Tracking Data (Occupancy, Tracks, Transitions) ---
            const trackingData = cameraData.tracking;
            if (trackingData && typeof trackingData === 'object') {
                
                // --- Occupancy (Aggregated across cameras) ---
                // Note: Occupancy might be better handled globally if zones span cameras.
                // For now, we'll display based on the first camera reporting it or aggregate.
                if (trackingData.occupancy && typeof trackingData.occupancy === 'object') {
                    for (const zone in trackingData.occupancy) {
                        globalOccupancy[zone] = (globalOccupancy[zone] || 0) + trackingData.occupancy[zone];
                    }
                }
                
                // --- Active Tracks (Aggregated across cameras) ---
                if (trackingData.active_tracks && Array.isArray(trackingData.active_tracks)) {
                    allActiveTracks = allActiveTracks.concat(trackingData.active_tracks);
                }

                // --- Transitions (Assuming transitions are also per-camera in trackingData) ---
                if (trackingData.transitions && Array.isArray(trackingData.transitions)) {
                     allTransitions = allTransitions.concat(trackingData.transitions);
                }
            }
        } // End loop through cameras

        // --- Update UI with Aggregated Data ---

        // Update Occupancy Display
        if (occupancyContentElem) {
            let occupancyHTML = '<ul style="margin: 0; padding-left: 15px;">';
            const sortedZones = Object.entries(globalOccupancy).sort(([, a], [, b]) => b - a);
            if (sortedZones.length > 0) {
                sortedZones.forEach(([zone, count]) => {
                    occupancyHTML += `<li><strong>${zone}:</strong> ${count}</li>`;
                });
            } else {
                occupancyHTML += '<li>No occupancy data.</li>'; // Updated message
            }
            occupancyHTML += '</ul>';
            occupancyContentElem.innerHTML = occupancyHTML;
        }

        // Update Active Tracks Display
        if (trackDetailsContentElem) {
            let tracksHTML = '';
            if (allActiveTracks.length > 0) {
                // Sort tracks by ID (ensure track_id is present)
                allActiveTracks.sort((a, b) => (a.track_id || 0) - (b.track_id || 0)).forEach(track => {
                    tracksHTML += `<div class="track-detail-item">`;
                    tracksHTML += `<strong>ID ${track.track_id || 'N/A'} (${track.camera_id || 'N/A'}):</strong><br>`; // Use safe access
                    tracksHTML += `Zone: ${track.zone || '-'}, Dwell: ${track.dwell_time?.toFixed(1) ?? '0.0'}s<br>`;
                    const center = track.center || ['N/A', 'N/A'];
                    const velocity = track.velocity || [0, 0];
                    const speed = Math.sqrt(velocity[0]**2 + velocity[1]**2);
                    tracksHTML += `Pos: [${center[0]}, ${center[1]}], Speed: ${speed?.toFixed(1) ?? '0.0'} px/s`;
                    tracksHTML += `</div>`;
                });
            } else {
                tracksHTML = '<span>No active tracks.</span>';
            }
            trackDetailsContentElem.innerHTML = tracksHTML;
        }

        // Update Transitions Display
        if (transitionsListElem) {
            let transitionsHTML = '';
            if (allTransitions.length > 0) {
                 // Sort by timestamp descending (assuming timestamp exists)
                 allTransitions.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0))
                     .slice(0, 10) // Limit to latest 10
                     .forEach(t => {
                    const timeStr = t.timestamp ? new Date(t.timestamp * 1000).toLocaleTimeString() : '??:??:??';
                    transitionsHTML += `<li>[${timeStr}] ID ${t.track_id || 'N/A'} (${t.camera_id || 'N/A'}): ${t.from_zone || 'N/A'} → ${t.to_zone || 'N/A'}</li>`;
                });
            } else {
                transitionsHTML = '<li>No recent transitions.</li>';
            }
            transitionsListElem.innerHTML = transitionsHTML;
        }

    } else {
        // Handle case where payload.cameras is missing or not an object
        console.warn("Stats payload missing or invalid 'cameras' structure.");
        // Optionally reset parts of the UI
        resetStatsDisplay(); // Or reset specific parts like performance
    }

    // --- REMOVED Old Logic that assumed top-level keys --- 
    // Performance
    // if (payload.performance) { ... }

    // System Status
    // if (payload.system_status && systemStatusElem) { ... }

    // Occupancy
    // if (payload.occupancy && occupancyContentElem) { ... }

    // Active Tracks
    // if (payload.tracks && trackDetailsContentElem) { ... }

    // Transitions
    // if (payload.transitions && transitionsListElem) { ... }
}

// --- Clock Update Function ---
function updateClock() {
    const now = new Date();
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    if (clockElem) {
        clockElem.textContent = `${hours}:${minutes}:${seconds}`;
    }
}

// Function to reset stats display to initial state
function resetStatsDisplay() {
    if (livingRoomPerfElem) livingRoomPerfElem.textContent = 'FPS: 0.0';
    if (kitchenPerfElem) kitchenPerfElem.textContent = 'FPS: 0.0';
    if (occupancyContentElem) occupancyContentElem.innerHTML = 'Loading...';
    if (trackDetailsContentElem) trackDetailsContentElem.innerHTML = 'Loading...';
    if (transitionsListElem) transitionsListElem.innerHTML = '<li>Loading...</li>';
    if (systemStatusElem) systemStatusElem.textContent = 'Uptime: 0s';
}

// --- Fullscreen Logic ---
fullscreenBtns.forEach(button => {
    button.addEventListener('click', () => {
        const targetId = button.getAttribute('data-target');
        const targetElement = document.getElementById(targetId);
        
        if (targetElement) {
            // Check if the element is already maximized
            const isMaximized = targetElement.classList.contains('maximized-stream');
            
            // Find the performance stats element
            let perfStatsElem = null;
            if (targetId === 'kitchen-stream') {
                perfStatsElem = kitchenPerfElem;
            } else if (targetId === 'living-room-stream') {
                perfStatsElem = livingRoomPerfElem;
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
        } else {
            console.error('Fullscreen target element not found:', targetId);
        }
    });
});

// Add event listener for the clear stats button
if (clearStatsButton) {
    clearStatsButton.addEventListener('click', () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log('Sending clear_stats message to backend...');
            socket.send(JSON.stringify({ type: 'clear_stats' }));
        } else {
            console.warn('WebSocket not connected. Cannot send clear_stats message.');
        }
    });
}

// Add keyboard event listener for escape key
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        // Remove maximized class from all video streams
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
    }
});

// --- ADDED: Function to handle toggle changes ---
// Function removed since toggle UI elements were removed
/* 
function handleVisToggleChange(event) {
    const toggleName = event.target.id.replace('toggle-', '').replace('-', '_'); // e.g., 'toggle-show-boxes' -> 'show_boxes'
    const isEnabled = event.target.checked;

    if (socket && socket.readyState === WebSocket.OPEN) {
        const message = {
            type: 'set_vis_toggle',
            toggle_name: toggleName,
            enabled: isEnabled
        };
        console.log(`Sending vis toggle message:`, message);
        socket.send(JSON.stringify(message));
    } else {
        console.warn('WebSocket not connected. Cannot send vis toggle message.');
        // Optional: Revert checkbox state if connection lost?
        // event.target.checked = !isEnabled;
    }
}
*/

// --- ADDED: Detection Configuration Sync Functions ---

function syncDetectionConfig(config) {
    console.log('[syncDetectionConfig] Syncing detection config:', config);
    
    // Update confidence threshold
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const confidenceValue = document.getElementById('confidence-value');
    if (confidenceThreshold && confidenceValue && config.confidence_threshold !== undefined) {
        confidenceThreshold.value = config.confidence_threshold;
        confidenceValue.textContent = config.confidence_threshold.toFixed(2);
    }
    
    // Update IOU threshold
    const iouThreshold = document.getElementById('iou-threshold');
    const iouValue = document.getElementById('iou-value');
    if (iouThreshold && iouValue && config.iou_threshold !== undefined) {
        iouThreshold.value = config.iou_threshold;
        iouValue.textContent = config.iou_threshold.toFixed(2);
    }
    
    // Update detection enabled toggle
    const toggleDetectionEnabled = document.getElementById('toggle-detection-enabled');
    if (toggleDetectionEnabled && config.detection_enabled !== undefined) {
        toggleDetectionEnabled.checked = config.detection_enabled;
    }
    
    // Update detection type toggles
    if (config.detection_toggles) {
        const toggleDetectPeople = document.getElementById('toggle-detect-people');
        if (toggleDetectPeople && config.detection_toggles.detect_people !== undefined) {
            toggleDetectPeople.checked = config.detection_toggles.detect_people;
        }
        
        const toggleDetectVehicles = document.getElementById('toggle-detect-vehicles');
        if (toggleDetectVehicles && config.detection_toggles.detect_vehicles !== undefined) {
            toggleDetectVehicles.checked = config.detection_toggles.detect_vehicles;
        }
        
        const toggleDetectFurniture = document.getElementById('toggle-detect-furniture');
        if (toggleDetectFurniture && config.detection_toggles.detect_furniture !== undefined) {
            toggleDetectFurniture.checked = config.detection_toggles.detect_furniture;
        }
    }
}

function syncDetectionToggle(toggleName, enabled) {
    console.log('[syncDetectionToggle] Syncing detection toggle:', toggleName, enabled);
    
    const toggleElement = document.getElementById(`toggle-${toggleName.replace('_', '-')}`);
    if (toggleElement) {
        toggleElement.checked = enabled;
    }
}

// --- Initial Setup --- Run after DOM is loaded ---
document.addEventListener('DOMContentLoaded', () => {
    console.log('Renderer DOM fully loaded and parsed');

    // Log the presence of stream image elements
    console.log('DOM Elements check:');
    console.log('- Living Room Stream Element:', !!livingRoomStreamElem);
    console.log('- Kitchen Stream Element:', !!kitchenStreamElem);

    // Add load event listeners to image elements
    if (livingRoomStreamElem) {
        livingRoomStreamElem.addEventListener('load', function() {
            // console.log('Living Room image loaded successfully'); // Commented out
        });
    }
    if (kitchenStreamElem) {
        kitchenStreamElem.addEventListener('load', function() {
            // console.log('Kitchen image loaded successfully'); // Commented out
        });
    }

    // Initial UI states
    resetStatsDisplay();
    updateClock();
    setInterval(updateClock, 1000); // Start clock interval

    // --- ADDED: Attach event listeners to toggles --- 
    // Add keypoints toggle
    const toggleShowKeypoints = document.getElementById('toggle-show-keypoints');
    if (toggleShowKeypoints) {
        toggleShowKeypoints.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_vis_toggle',
                    toggle_name: 'show_keypoints',
                    enabled: event.target.checked
                };
                socket.send(JSON.stringify(message));
            }
        });
    }

    // Add mask toggle
    const toggleShowMasks = document.getElementById('toggle-show-masks');
    if (toggleShowMasks) {
        toggleShowMasks.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_vis_toggle',
                    toggle_name: 'show_masks',
                    enabled: event.target.checked
                };
                socket.send(JSON.stringify(message));
            }
        });
    }

    // --- ADDED: Detection Controls ---
    
    // Detection type toggles
    const toggleDetectPeople = document.getElementById('toggle-detect-people');
    if (toggleDetectPeople) {
        toggleDetectPeople.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_toggle',
                    toggle_name: 'detect_people',
                    enabled: event.target.checked
                };
                console.log('Sending detection toggle message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }
    
    const toggleDetectVehicles = document.getElementById('toggle-detect-vehicles');
    if (toggleDetectVehicles) {
        toggleDetectVehicles.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_toggle',
                    toggle_name: 'detect_vehicles',
                    enabled: event.target.checked
                };
                console.log('Sending detection toggle message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }
    
    const toggleDetectFurniture = document.getElementById('toggle-detect-furniture');
    if (toggleDetectFurniture) {
        toggleDetectFurniture.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_toggle',
                    toggle_name: 'detect_furniture',
                    enabled: event.target.checked
                };
                console.log('Sending detection toggle message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }
    
    // Detection settings sliders
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const confidenceValue = document.getElementById('confidence-value');
    if (confidenceThreshold && confidenceValue) {
        confidenceThreshold.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            confidenceValue.textContent = value.toFixed(2);
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_config',
                    config: {
                        confidence_threshold: value
                    }
                };
                console.log('Sending detection config message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }
    
    const iouThreshold = document.getElementById('iou-threshold');
    const iouValue = document.getElementById('iou-value');
    if (iouThreshold && iouValue) {
        iouThreshold.addEventListener('input', (event) => {
            const value = parseFloat(event.target.value);
            iouValue.textContent = value.toFixed(2);
            
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_config',
                    config: {
                        iou_threshold: value
                    }
                };
                console.log('Sending detection config message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }
    
    // Detection enable/disable toggle
    const toggleDetectionEnabled = document.getElementById('toggle-detection-enabled');
    if (toggleDetectionEnabled) {
        toggleDetectionEnabled.addEventListener('change', (event) => {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const message = {
                    type: 'set_detection_config',
                    config: {
                        detection_enabled: event.target.checked
                    }
                };
                console.log('Sending detection config message:', message);
                socket.send(JSON.stringify(message));
            }
        });
    }

    // --- ADDED: Collapsible Controls Section ---
    const controlsHeader = document.getElementById('controls-header');
    const controlsContent = document.getElementById('controls-content');
    const controlsSection = document.querySelector('.controls-section');
    const collapseToggle = document.getElementById('controls-collapse-toggle');
    
    if (controlsHeader && controlsContent && controlsSection && collapseToggle) {
        // Initialize state (collapsed by default)
        let isCollapsed = true;
        
        // Function to toggle collapse state
        function toggleCollapse() {
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
        }
        
        // Apply initial collapsed state (no need to add collapsed class since it's default)
        controlsSection.classList.add('collapsed');
        collapseToggle.textContent = '+';
        
                // Add click event listeners
                controlsHeader.addEventListener('click', (event) => {
                    // Don't trigger if clicking on the toggle button itself
                    if (event.target !== collapseToggle && !collapseToggle.contains(event.target)) {
                        toggleCollapse();
                    }
                });
                
                collapseToggle.addEventListener('click', (event) => {
                    event.stopPropagation(); // Prevent header click from also firing
                    toggleCollapse();
                });
    }

    // Start WebSocket connection
    connectWebSocket();
});