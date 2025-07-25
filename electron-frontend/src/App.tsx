import React, { useEffect, useState } from 'react';
import './style.css';
import { TelemetryProvider } from './telemetry/TelemetryContext';
import { TelemetryDrawer } from './telemetry/TelemetryDrawer';
import { TelemetryToggle } from './telemetry/TelemetryToggle';

const Dashboard: React.FC = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);

  useEffect(() => {
    // legacy renderer logic
    const script = document.createElement('script');
    script.src = './renderer.js';
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);


  return (
    <div id="app-container">
      <header>
        <h1>Smart Room Dashboard</h1>
        <div id="system-status">Uptime: 0s</div>
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
            <div id="kitchen-perf" className="perf-stats">FPS: 0.0</div>
          </div>

          <div className="video-container">
            <h2>Living Room</h2>
            <div className="video-wrapper">
              <img id="living-room-stream" src="" alt="Living Room Stream" width="640" height="360" />
              <button className="fullscreen-btn" data-target="living-room-stream">Max</button>
            </div>
            <div id="living-room-perf" className="perf-stats">FPS: 0.0</div>
          </div>
        </div>

        <div className="stats-section">
          <div className="stats-card" id="occupancy-card">
            <h3>Zone Occupancy</h3>
            <div id="occupancy-content">Loading...</div>
          </div>
          <div className="stats-card" id="track-details-card">
            <h3>Active Tracks</h3>
            <div id="track-details-content">Loading...</div>
          </div>
          <div className="stats-card" id="transitions-card">
            <h3>Recent Transitions</h3>
            <ul id="transitions-list"><li>Loading...</li></ul>
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
        <span id="connection-status">Status: Disconnected</span>
        <button id="clear-stats-btn" className="footer-button">Clear Stats</button>
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

