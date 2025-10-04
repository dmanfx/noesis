import React, { useMemo, useState } from 'react';
import { useTelemetry } from './TelemetryContext';
import { CameraKey, cameraLabel } from '../lib/camera';

interface TelemetryPanelProps {
  onClose?: () => void;
  cameraStatuses: Record<CameraKey, string>;
}

const CAMERA_ORDER: CameraKey[] = ['living-room', 'kitchen', 'family-room'];

const statusToVisual = (statusRaw: string | undefined) => {
  const status = (statusRaw || '').toLowerCase();
  if (!status) {
    return { icon: '?', color: 'var(--muted)' };
  }
  if (status.includes('run') || status.includes('ok') || status.includes('online') || status.includes('ready')) {
    return { icon: '✓', color: '#39d98a' };
  }
  if (status.includes('error') || status.includes('fail') || status.includes('offline') || status.includes('stop')) {
    return { icon: '✕', color: '#e76a6a' };
  }
  return { icon: '?', color: 'var(--muted)' };
};

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({ onClose, cameraStatuses }) => {
  const { entries } = useTelemetry();
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(e => `${e.group} ${e.key}`.toLowerCase().includes(q));
  }, [entries, query]);

  const sorted = useMemo(() => [...filtered].sort((a, b) => a.group.localeCompare(b.group) || a.key.localeCompare(b.key)), [filtered]);

  return (
    <div className="telemetry-panel">
      <div className="telemetry-header">
        <strong>Telemetry</strong>
        <div className="spacer" />
        <div className="search-wrap">
          <input
            className="search-input"
            placeholder="Filter..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button
              className="clear-btn"
              aria-label="Clear filter"
              title="Clear"
              onClick={() => setQuery('')}
            >×</button>
          )}
        </div>
        <button className="btn" onClick={onClose}>Hide</button>
      </div>
      <div className="telemetry-body">
        <div className="camera-status-row">
          {CAMERA_ORDER.map((key) => {
            const visual = statusToVisual(cameraStatuses[key]);
            return (
              <div key={key} className="camera-status-card" title={cameraStatuses[key] || 'unknown'}>
                <div className="camera-status-label">{cameraLabel(key)}</div>
                <div className="camera-status-icon" style={{ color: visual.color }}>{visual.icon}</div>
              </div>
            );
          })}
        </div>
        {sorted.map((e) => (
          <div key={`${e.group}:${e.key}`} className="telemetry-item">
            <div className="mono" style={{ color: '#9db1c8' }}>{e.group}</div>
            <div>{e.key}</div>
            <div className="mono" style={{ justifySelf: 'end' }}>{String(e.value)}</div>
          </div>
        ))}
        {sorted.length === 0 && <div style={{ color: 'var(--muted)', padding: 8 }}>No telemetry yet.</div>}
      </div>
    </div>
  );
};
