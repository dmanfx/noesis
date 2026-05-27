import React, { useMemo, useState } from 'react';
import { useTelemetry, TelemetryEntry } from './TelemetryContext';
import { CameraKey, cameraLabel, cameraOrder } from '../lib/camera';

interface TelemetryPanelProps {
  onClose?: () => void;
  cameraStatuses: Record<CameraKey, string>;
  cameraPoses?: Record<CameraKey, {
    x: number;
    y: number;
    z: number;
    heightAboveFloor?: number | null;
    forwardFx?: number | null;
    forwardFz?: number | null;
  } | null>;
}

type MapAnythingSection = {
  name: string;
  general: TelemetryEntry[];
  perCam: Record<CameraKey, TelemetryEntry[]>;
};

const CAMERA_LABEL_TO_KEY: Record<string, CameraKey> = cameraOrder.reduce((acc, key) => {
  acc[cameraLabel(key)] = key;
  return acc;
}, {} as Record<string, CameraKey>);

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

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({ onClose, cameraStatuses, cameraPoses }) => {
  const { entries } = useTelemetry();
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(e => `${e.group} ${e.key}`.toLowerCase().includes(q));
  }, [entries, query]);

  const { mapAnythingSections, otherEntries } = useMemo(() => {
    const sections = new Map<string, MapAnythingSection>();
    const others: TelemetryEntry[] = [];

    const initPerCam = (): Record<CameraKey, TelemetryEntry[]> => {
      return cameraOrder.reduce((acc, key) => {
        acc[key] = [];
        return acc;
      }, {} as Record<CameraKey, TelemetryEntry[]>);
    };

    // Debug-only keys from the calibration bundle / MapAnything that are not meant for the normal dashboard view.
    // These were appearing as blank boxes (K, E, Pose, pose confidence, etc.) underneath the BEV.
    const DEBUG_ONLY_KEYS = new Set([
      'K', 'E', 'Pose', 'pose confidence', 'pose_confidence',
      'intrinsics', 'extrinsics', 'pose', 'confidence',
    ]);

    for (const entry of filtered) {
      const keyLower = entry.key.toLowerCase().trim();
      if (DEBUG_ONLY_KEYS.has(entry.key) || DEBUG_ONLY_KEYS.has(keyLower)) {
        continue; // hide raw debug calibration fields
      }

      if (!entry.group.startsWith('MapAnything')) {
        others.push(entry);
        continue;
      }

      let section = sections.get(entry.group);
      if (!section) {
        section = {
          name: entry.group,
          general: [],
          perCam: initPerCam(),
        };
        sections.set(entry.group, section);
      }

      let cameraKey: CameraKey | null = null;
      let displayKey = entry.key;
      for (const [label, key] of Object.entries(CAMERA_LABEL_TO_KEY)) {
        if (entry.key.startsWith(label)) {
          cameraKey = key;
          const remainder = entry.key.slice(label.length).trim();
          if (remainder) {
            displayKey = remainder;
          }
          break;
        }
      }

      const storedEntry = { ...entry, key: displayKey };
      if (cameraKey) {
        section.perCam[cameraKey].push(storedEntry);
      } else {
        section.general.push(storedEntry);
      }
    }

    const sectionsArray = Array.from(sections.values());
    const sortedOthers = [...others].sort((a, b) => a.group.localeCompare(b.group) || a.key.localeCompare(b.key));
    return { mapAnythingSections: sectionsArray, otherEntries: sortedOthers };
  }, [filtered]);

  const hasTelemetryEntries = mapAnythingSections.length > 0 || otherEntries.length > 0;

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
          {cameraOrder.map((key) => {
            const visual = statusToVisual(cameraStatuses[key]);
            return (
              <div key={key} className="camera-status-card" title={cameraStatuses[key] || 'unknown'}>
                <div className="camera-status-label">{cameraLabel(key)}</div>
                <div className="camera-status-icon" style={{ color: visual.color }}>{visual.icon}</div>
              </div>
            );
          })}
        </div>
        {cameraPoses && (
          <div style={{ marginBottom: 10, fontSize: 11 }}>
            {cameraOrder.map((key) => {
              const pose = cameraPoses[key];
              if (!pose || !Number.isFinite(pose.x) || !Number.isFinite(pose.z)) return null;
              const height = Number.isFinite(pose.heightAboveFloor ?? NaN)
                ? pose.heightAboveFloor
                : pose.y;
              let heading: number | null = null;
              if (Number.isFinite(pose.forwardFx ?? NaN) && Number.isFinite(pose.forwardFz ?? NaN)) {
                heading = Math.atan2(pose.forwardFz as number, pose.forwardFx as number) * 180 / Math.PI;
              }
              return (
                <div key={`pose-${key}`} className="telemetry-item" style={{ borderBottom: 'none', paddingTop: 2, paddingBottom: 2 }}>
                  <div className="mono" style={{ color: '#9db1c8' }}>{cameraLabel(key)}</div>
                  <div>
                    Pos: [{pose.x.toFixed(2)}, {pose.z.toFixed(2)}] m
                    {Number.isFinite(height ?? NaN) && (
                      <span> · Height: {Number(height).toFixed(2)} m</span>
                    )}
                  </div>
                  <div className="mono" style={{ justifySelf: 'end' }}>
                    {heading !== null ? `${heading.toFixed(0)}°` : ''}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {mapAnythingSections.map((section) => (
          <div key={`section-${section.name}`} className="telemetry-section">
            <div className="telemetry-section-title">{section.name}</div>
            {cameraOrder.map((key) => {
              const rows = section.perCam[key];
              if (!rows || !rows.length) return null;
              return (
                <div key={`${section.name}-${key}`} className="telemetry-subsection">
                  <div className="telemetry-subsection-title">{cameraLabel(key)}</div>
                  <div className="telemetry-subsection-body">
                    {rows.map((row) => (
                      <div key={`${section.name}-${key}-${row.key}`} className="telemetry-data-row">
                        <span>{row.key}</span>
                        <span className="mono" style={{ color: '#9db1c8' }}>{String(row.value)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
            {section.general.length > 0 && (
              <div className="telemetry-subsection">
                <div className="telemetry-subsection-title">General</div>
                <div className="telemetry-subsection-body">
                  {section.general.map((row) => (
                    <div key={`${section.name}-general-${row.key}`} className="telemetry-data-row">
                      <span>{row.key}</span>
                      <span className="mono" style={{ color: '#9db1c8' }}>{String(row.value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
        {otherEntries.map((e) => (
          <div key={`${e.group}:${e.key}`} className="telemetry-item">
            <div className="mono" style={{ color: '#9db1c8' }}>{e.group}</div>
            <div>{e.key}</div>
            <div className="mono" style={{ justifySelf: 'end' }}>{String(e.value)}</div>
          </div>
        ))}
        {!hasTelemetryEntries && <div style={{ color: 'var(--muted)', padding: 8 }}>No telemetry yet.</div>}
      </div>
    </div>
  );
};
