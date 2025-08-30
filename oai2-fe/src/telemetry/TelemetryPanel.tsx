import React, { useMemo, useState } from 'react';
import { useTelemetry } from './TelemetryContext';

export const TelemetryPanel: React.FC<{ onClose?: () => void }> = ({ onClose }) => {
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
