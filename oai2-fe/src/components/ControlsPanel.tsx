import React, { useEffect, useState } from 'react';

export const ControlsPanel: React.FC<{
  trailEnabled: boolean;
  setTrailEnabled: (v: boolean) => void;
  onSendTrail: (v: boolean) => void;
  onClearStats: () => void;
  onSendDetectionConfig: (config: any) => void;
  onSendDetectionToggle: (name: string, enabled: boolean) => void;
}> = ({ trailEnabled, setTrailEnabled, onSendTrail, onClearStats, onSendDetectionConfig, onSendDetectionToggle }) => {
  const [people, setPeople] = useState(true);
  const [vehicles, setVehicles] = useState(false);
  const [furniture, setFurniture] = useState(false);
  const [conf, setConf] = useState(0.3);
  const [iou, setIou] = useState(0.45);

  useEffect(() => {
    // publish config on change (debounced)
    const h = setTimeout(() => onSendDetectionConfig({ confidence: conf, iou }), 250);
    return () => clearTimeout(h);
  }, [conf, iou]);

  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="panel card vstack">
      <div className="hstack" style={{ marginBottom: collapsed ? 0 : 8 }}>
        <div className="card-title">Detection Controls</div>
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
          <div className="list">
            <div className="row">
              <label className="grow">Detect People</label>
              <input type="checkbox" checked={people} onChange={(e) => { setPeople(e.target.checked); onSendDetectionToggle('detect_people', e.target.checked); }} />
            </div>
            <div className="row">
              <label className="grow">Detect Vehicles</label>
              <input type="checkbox" checked={vehicles} onChange={(e) => { setVehicles(e.target.checked); onSendDetectionToggle('detect_vehicles', e.target.checked); }} />
            </div>
            <div className="row">
              <label className="grow">Detect Furniture</label>
              <input type="checkbox" checked={furniture} onChange={(e) => { setFurniture(e.target.checked); onSendDetectionToggle('detect_furniture', e.target.checked); }} />
            </div>
          </div>

          <div className="vstack" style={{ marginTop: 8 }}>
            <label>Confidence Threshold <span className="mono pill">{conf.toFixed(2)}</span></label>
            <input type="range" min={0.1} max={0.9} step={0.05} value={conf} onChange={(e) => setConf(parseFloat(e.target.value))} />
            <label>IOU Threshold <span className="mono pill">{iou.toFixed(2)}</span></label>
            <input type="range" min={0.1} max={0.9} step={0.05} value={iou} onChange={(e) => setIou(parseFloat(e.target.value))} />
          </div>

          <div className="vstack" style={{ marginTop: 8 }}>
            <label className="row"><span className="grow">Enable Object Trails</span> <input type="checkbox" checked={trailEnabled} onChange={(e) => { setTrailEnabled(e.target.checked); onSendTrail(e.target.checked); }} /></label>
          </div>

          <div className="hstack" style={{ marginTop: 10 }}>
            <button className="btn danger" onClick={onClearStats}>Clear Stats</button>
            <div className="spacer" />
            <span className="subtitle">Changes broadcast to server</span>
          </div>
        </>
      )}
    </div>
  );
};
