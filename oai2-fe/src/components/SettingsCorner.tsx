import React, { useEffect, useId, useState } from 'react';
import '../styles/settings-corner.css';

type Props = {
  connected: boolean;
  calibrating: boolean;
  onCalibrate: () => void;
};

const SettingsCorner: React.FC<Props> = ({ connected, calibrating, onCalibrate }) => {
  const [open, setOpen] = useState(false);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open]);

  const calibrateDisabled = !connected || calibrating;
  const calibrateTitle = !connected
    ? 'Connect to enable calibration'
    : calibrating
      ? 'Calibration in progress'
      : 'Auto-calibrate poses from latest depth';

  return (
    <>
      <div
        className={open ? 'settings-overlay open' : 'settings-overlay'}
        onClick={() => setOpen(false)}
        aria-hidden={!open}
      />
      <div className="settings-corner">
        <div
          id={panelId}
          className={open ? 'settings-panel open' : 'settings-panel'}
          aria-hidden={!open}
          role="dialog"
          aria-label="Settings"
        >
          <div className="settings-header">
            <div className="settings-title">Settings</div>
            <button
              className="settings-close"
              type="button"
              aria-label="Close settings"
              title="Close"
              onClick={() => setOpen(false)}
            >
              ×
            </button>
          </div>

          <div className="settings-body">
            <div className="settings-row">
              <div className="settings-row-text">
                <div className="settings-row-title">Calibration</div>
                <div className="settings-row-subtitle">Auto-calibrate poses from latest depth</div>
              </div>
              <button
                className="btn ghost settings-action"
                type="button"
                disabled={calibrateDisabled}
                title={calibrateTitle}
                onClick={() => {
                  onCalibrate();
                  setOpen(false);
                }}
              >
                {calibrating ? 'Running...' : 'Calibrate'}
              </button>
            </div>
          </div>
        </div>

        <button
          className={open ? 'settings-fab open' : 'settings-fab'}
          type="button"
          aria-label="Settings"
          aria-expanded={open}
          aria-controls={panelId}
          title="Settings"
          onClick={() => setOpen(v => !v)}
        >
          <svg aria-hidden="true" viewBox="0 0 24 24" role="presentation">
            <path
              fill="currentColor"
              d="M12 8.9a3.1 3.1 0 1 0 0 6.2 3.1 3.1 0 0 0 0-6.2Zm8.1 3.1c0-.4 0-.8-.1-1.2l2-1.6-1.9-3.3-2.4 1a7.9 7.9 0 0 0-2.1-1.2l-.4-2.6h-3.8l-.4 2.6c-.8.2-1.5.6-2.1 1.2l-2.4-1-1.9 3.3 2 1.6a8.6 8.6 0 0 0 0 2.4l-2 1.6 1.9 3.3 2.4-1c.6.5 1.3.9 2.1 1.2l.4 2.6h3.8l.4-2.6c.8-.2 1.5-.6 2.1-1.2l2.4 1 1.9-3.3-2-1.6c.1-.4.1-.8.1-1.2Z"
            />
          </svg>
        </button>
      </div>
    </>
  );
};

export default SettingsCorner;
