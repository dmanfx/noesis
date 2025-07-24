import React, { useMemo, useState } from 'react';
import styled from 'styled-components';
import { useTelemetry, TelemetryEntry } from './TelemetryContext';

const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.3);
  z-index: 999;
`;

const Drawer = styled.div<{open: boolean}>`
  position: fixed;
  top: 0;
  right: 0;
  width: 320px;
  height: 100vh;
  background: #2b2b2b;
  box-shadow: -2px 0 8px rgba(0,0,0,0.4);
  transition: transform 0.3s ease;
  transform: translateX(${props => (props.open ? '0' : '100%')});
  z-index: 1000;
  display: flex;
  flex-direction: column;
`;

const Header = styled.div`
  padding: 10px;
  border-bottom: 1px solid #444;
  font-weight: bold;
`;

const SearchInput = styled.input`
  margin: 10px;
  padding: 6px;
  border-radius: 4px;
  border: 1px solid #555;
  background: #1e1e1e;
  color: #fff;
`;

const GroupHeader = styled.div`
  padding: 4px 10px;
  background: #383838;
  font-weight: bold;
`;

const Row = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 4px 10px;
  font-size: 0.9em;
  &:hover {
    background: #444;
  }
`;

const Value = styled.span`
  font-family: monospace;
`;

interface DrawerProps {
  open: boolean;
  onClose: () => void;
}

export const TelemetryDrawer: React.FC<DrawerProps> = ({open, onClose}) => {
  const { entries } = useTelemetry();
  const [filter, setFilter] = useState('');

  const grouped = useMemo(() => {
    const map: Record<string, TelemetryEntry[]> = {};
    entries.forEach(e => {
      if (filter && !e.key.toLowerCase().includes(filter.toLowerCase())) return;
      if (!map[e.group]) map[e.group] = [];
      map[e.group].push(e);
    });
    return map;
  }, [entries, filter]);

  const now = Date.now();

  return (
    <>
      {open && <Overlay onClick={onClose} />}
      <Drawer open={open}>
        <Header>Telemetry</Header>
        <SearchInput
          placeholder="Filter keys..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
        />
        <div style={{overflowY: 'auto', flex: 1}}>
          {Object.entries(grouped).map(([group, items]) => (
            <div key={group}>
              <GroupHeader>{group}</GroupHeader>
              {items.map(item => (
                <Row key={`${group}-${item.key}`}>
                  <span>{item.key}</span>
                  <Value>{item.value}</Value>
                  <span style={{marginLeft: '8px', fontSize: '0.8em', opacity: 0.6}}>
                    {Math.round((now - item.ts)/1000)}s ago
                  </span>
                </Row>
              ))}
            </div>
          ))}
        </div>
      </Drawer>
    </>
  );
};
