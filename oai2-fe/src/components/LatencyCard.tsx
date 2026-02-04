import React, { useMemo } from 'react';
import { CameraKey, cameraLabel, cameraOrder } from '../lib/camera';
import { LatencyMetrics } from '../types/latency';

const fmtMs = (v: unknown): string => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  return `${n.toFixed(1)} ms`;
};

const fmtSec = (v: unknown): string => {
  const n = Number(v);
  if (!Number.isFinite(n)) return '—';
  if (n < 1.0) return `${Math.round(n * 1000)} ms`;
  return `${n.toFixed(1)} s`;
};

export const LatencyCard: React.FC<{
  pipeline?: LatencyMetrics | null;
  perCam?: Partial<Record<CameraKey, LatencyMetrics | null>> | null;
}> = ({ pipeline, perCam }) => {
  const pipelineText = useMemo(() => {
    if (!pipeline) return { title: 'No data', subtitle: 'Waiting for stats…' };
    if (pipeline.enabled === false) {
      return { title: 'Disabled', subtitle: pipeline.reason ? String(pipeline.reason) : 'NVDS latency measurement not enabled' };
    }
    const count = Number(pipeline.count ?? 0) || 0;
    if (count <= 0) return { title: 'No samples yet', subtitle: `window ${(pipeline.window_sec ?? 0) || 10}s` };
    const age = pipeline.last_sample_age_sec;
    const ageTxt = age == null ? '' : ` · last ${fmtSec(age)} ago`;
    return { title: `p95 ${fmtMs(pipeline.p95)}`, subtitle: `p50 ${fmtMs(pipeline.p50)} · max ${fmtMs(pipeline.max)}${ageTxt}` };
  }, [pipeline]);

  return (
    <div className="panel card">
      <div className="card-title">Latency</div>
      <div className="latency-summary">
        <div className="latency-summary-main">{pipelineText.title}</div>
        <div className="latency-summary-sub">{pipelineText.subtitle}</div>
      </div>
      <div className="list" style={{ marginTop: 10 }}>
        {cameraOrder.map((key) => {
          const m = perCam?.[key] ?? null;
          if (!m) {
            return (
              <div key={`lat-${key}`} className="row" style={{ justifyContent: 'space-between' }}>
                <span>{cameraLabel(key)}</span>
                <span className="mono" style={{ color: 'var(--muted)' }}>—</span>
              </div>
            );
          }
          if (m.enabled === false) {
            return (
              <div key={`lat-${key}`} className="row" style={{ justifyContent: 'space-between' }}>
                <span>{cameraLabel(key)}</span>
                <span className="mono" style={{ color: 'var(--muted)' }}>disabled</span>
              </div>
            );
          }
          const count = Number(m.count ?? 0) || 0;
          const right = count > 0 ? fmtMs(m.p95) : '—';
          return (
            <div key={`lat-${key}`} className="row" style={{ justifyContent: 'space-between' }}>
              <span>{cameraLabel(key)}</span>
              <span className="mono pill">{right}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
