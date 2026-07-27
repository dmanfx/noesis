import React, { useState } from 'react';

export const ControlsPanel: React.FC<{
  trailEnabled: boolean;
  setTrailEnabled: (v: boolean) => void;
  onSendTrail: (v: boolean) => void;
  onClearStats: () => void;
}> = ({ trailEnabled, setTrailEnabled, onSendTrail, onClearStats }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="panel card vstack">
      <div className="hstack" style={{ marginBottom: collapsed ? 0 : 8 }}>
        <div className="card-title">Runtime Controls</div>
        <div className="spacer" />
        <button
          className="btn ghost"
          title={collapsed ? 'Expand' : 'Collapse'}
          aria-expanded={!collapsed}
          onClick={() => setCollapsed(v => !v)}
        >
          {collapsed ? '▸' : '▾'}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="vstack" style={{ marginTop: 8 }}>
            <label className="row"><span className="grow">Enable Object Trails</span> <input type="checkbox" checked={trailEnabled} onChange={(e) => { setTrailEnabled(e.target.checked); onSendTrail(e.target.checked); }} /></label>
          </div>

          <div className="hstack" style={{ marginTop: 10 }}>
            <button className="btn danger" onClick={onClearStats}>Clear Stats</button>
            <div className="spacer" />
            <span className="subtitle">Applied by the runtime</span>
          </div>
        </>
      )}
    </div>
  );
};
