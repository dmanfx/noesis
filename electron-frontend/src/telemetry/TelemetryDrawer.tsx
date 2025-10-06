import React, { useMemo, useState } from 'react';
import styled from 'styled-components';
import { TelemetryEntry, useTelemetry } from './TelemetryContext';

interface CollapsedSections {
  system: boolean;
  performance: boolean;
  activity: boolean;
  occupancy: boolean;
  details: boolean;
}

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
  width: 380px;
  height: 100vh;
  background: #2b2b2b;
  box-shadow: -2px 0 8px rgba(0,0,0,0.4);
  transition: transform 0.3s ease;
  transform: translateX(${props => (props.open ? '0' : '100%')});
  z-index: 1000;
  display: flex;
  flex-direction: column;
  font-size: 13px;
`;

const Header = styled.div`
  padding: 12px 16px;
  border-bottom: 1px solid #444;
  font-weight: bold;
  font-size: 14px;
  background: #1e1e1e;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const HeaderClock = styled.span`
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-weight: 500;
  font-size: 12px;
  color: #9aa0a6; /* subtle */
`;

const SearchInput = styled.input`
  margin: 12px 16px;
  padding: 8px 12px;
  border-radius: 4px;
  border: 1px solid #555;
  background: #1e1e1e;
  color: #fff;
  font-size: 12px;
  width: calc(100% - 32px);

  &:focus {
    outline: none;
    border-color: #007acc;
  }
`;

// Section headers with different styles for hierarchy
const SectionHeader = styled.div<{collapsible?: boolean}>`
  padding: 8px 16px;
  background: #383838;
  font-weight: bold;
  font-size: 13px;
  border-bottom: 1px solid #444;
  margin-top: 8px;
  cursor: ${props => props.collapsible ? 'pointer' : 'default'};
  display: flex;
  align-items: center;
  justify-content: space-between;

  &:hover {
    background: ${props => props.collapsible ? '#404040' : '#383838'};
  }
`;

const SectionToggle = styled.span<{collapsed: boolean}>`
  font-size: 12px;
  color: #ccc;
  transition: transform 0.2s ease;
  transform: ${props => props.collapsed ? 'rotate(0deg)' : 'rotate(90deg)'};
`;

const SubsectionHeader = styled.div`
  padding: 6px 16px;
  background: #2a2a2a;
  font-weight: bold;
  font-size: 12px;
  border-bottom: 1px solid #333;
  color: #ccc;
`;

const MetricRow = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 16px;
  font-size: 12px;
  border-bottom: 1px solid #222;
  transition: background-color 0.15s ease;

  &:hover {
    background: #333;
  }

  &:last-child {
    border-bottom: none;
  }
`;

const MetricLabel = styled.span`
  flex: 1;
  color: #ddd;
`;

const MetricValue = styled.span<{status?: 'good' | 'warning' | 'error'}>`
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-weight: 500;
  color: ${props => {
    switch (props.status) {
      case 'good': return '#4ade80';
      case 'warning': return '#fbbf24';
      case 'error': return '#f87171';
      default: return '#60a5fa';
    }
  }};
  text-align: right;
  min-width: 60px;
`;

const Timestamp = styled.span`
  font-size: 10px;
  color: #888;
  margin-left: 8px;
  min-width: 50px;
  text-align: right;
`;

// Compact grid for camera metrics
const CameraGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
  padding: 8px 16px;
`;

const CameraCard = styled.div`
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 8px;
  text-align: center;
`;

const CameraName = styled.div`
  font-size: 10px;
  color: #ccc;
  margin-bottom: 4px;
  text-transform: uppercase;
`;

const CameraValue = styled.div<{status?: 'good' | 'warning' | 'error'}>`
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-size: 13px;
  font-weight: bold;
  color: ${props => {
    switch (props.status) {
      case 'good': return '#4ade80';
      case 'warning': return '#fbbf24';
      case 'error': return '#f87171';
      default: return '#60a5fa';
    }
  }};
`;

const OccupancyGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 6px;
  padding: 8px 16px;
`;

const OccupancyItem = styled.div`
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 6px 8px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 2px;
`;

const ZoneName = styled.div`
  font-size: 11px;
  color: #ccc;
  font-weight: 500;
`;

const ZoneCount = styled.div`
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-size: 14px;
  font-weight: bold;
  color: #60a5fa;
`;

interface DrawerProps {
  open: boolean;
  onClose: () => void;
}

const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600).toString().padStart(2, '0');
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${h}h ${m}m ${s}s`;
};

const getStatusColor = (value: string | number, key: string): 'good' | 'warning' | 'error' | undefined => {
  if (typeof value === 'string') {
    const strValue = value.toLowerCase();
    if (strValue.includes('connected') || strValue.includes('running') || strValue.includes('yes')) return 'good';
    if (strValue.includes('disconnected') || strValue.includes('error') || strValue.includes('no')) return 'error';
  }
  return undefined;
};

const getCameraStatusColor = (value: string | number, key: string): 'good' | 'warning' | 'error' | undefined => {
  if (key === 'FPS' && typeof value === 'number') {
    if (value >= 25) return 'good';
    if (value >= 15) return 'warning';
    return 'error';
  }
  return undefined;
};

export const TelemetryDrawer: React.FC<DrawerProps> = ({open, onClose}) => {
  const { entries } = useTelemetry();
  const [filter, setFilter] = useState('');
  const [drawerClock, setDrawerClock] = useState('--:--:--');
  const [collapsedSections, setCollapsedSections] = useState<CollapsedSections>({
    system: false,
    performance: false,
    activity: false,
    occupancy: false,
    details: false
  });

  const toggleSection = (section: keyof CollapsedSections) => {
    setCollapsedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Organize telemetry data into logical sections
  const organizedData = useMemo(() => {
    const data = {
      system: [] as TelemetryEntry[],
      performance: {
        fps: [] as TelemetryEntry[],
        processing: [] as TelemetryEntry[],
        frames: [] as TelemetryEntry[]
      },
      activity: [] as TelemetryEntry[],
      occupancy: [] as TelemetryEntry[],
      reid: [] as TelemetryEntry[],
      details: [] as TelemetryEntry[]
    };

    entries.forEach(entry => {
      if (filter && !entry.key.toLowerCase().includes(filter.toLowerCase()) &&
          !entry.group.toLowerCase().includes(filter.toLowerCase())) return;

      // System health (most critical)
      if (entry.group === 'Connection' || entry.group === 'Application') {
        data.system.push(entry);
      }
      // Camera performance metrics - only include client-side FPS tracking (frontend players)
      else if (entry.group.startsWith('Camera ') &&
               entry.key === 'FPS' &&
               (entry.group === 'Camera Living Room' ||
                entry.group === 'Camera Kitchen' ||
                entry.group === 'Camera Family Room')) {
        data.performance.fps.push(entry);
      }
      else if (entry.group.startsWith('Camera ') && entry.key === 'Proc ms') {
        data.performance.processing.push(entry);
      }
      else if (entry.group.startsWith('Camera ') && entry.key === 'Frames') {
        data.performance.frames.push(entry);
      }
      // Activity metrics
      else if (entry.group === 'Tracking') {
        data.activity.push(entry);
      }
      // Occupancy data
      else if (entry.group === 'Occupancy') {
        data.occupancy.push(entry);
      }
      // ReID / StableID metrics
      else if (entry.group === 'ReID') {
        data.reid.push(entry);
      }
      // Everything else goes to details
      else {
        data.details.push(entry);
      }
    });

    return data;
  }, [entries, filter]);

  const now = Date.now();

  // Subtle header clock for the drawer
  React.useEffect(() => {
    const updateClock = () => {
      const d = new Date();
      const hh = String(d.getHours()).padStart(2, '0');
      const mm = String(d.getMinutes()).padStart(2, '0');
      const ss = String(d.getSeconds()).padStart(2, '0');
      setDrawerClock(`${hh}:${mm}:${ss}`);
    };
    updateClock();
    const id = setInterval(updateClock, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      {open && <Overlay onClick={onClose} />}
      <Drawer open={open}>
        <Header>
          <span>System Telemetry</span>
          <HeaderClock aria-label="Current time">{drawerClock}</HeaderClock>
        </Header>
        <SearchInput
          placeholder="Filter metrics..."
          value={filter}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setFilter(e.target.value)}
        />
        <div style={{overflowY: 'auto', flex: 1}}>
          {/* System Health Section */}
          {organizedData.system.length > 0 && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('system')}
              >
                <span>🔴 System Health</span>
                <SectionToggle collapsed={collapsedSections.system}>▶</SectionToggle>
              </SectionHeader>
              {!collapsedSections.system && organizedData.system.map(item => {
                const secondsAgo = Math.round((now - item.ts) / 1000);
                return (
                  <MetricRow key={`${item.group}-${item.key}`}>
                    <MetricLabel>{item.key}</MetricLabel>
                    <MetricValue status={getStatusColor(item.value, item.key)}>
                      {item.key === 'Uptime' ? formatUptime(item.value as number) : String(item.value)}
                    </MetricValue>
                    <Timestamp>{secondsAgo > 10 ? `${secondsAgo}s` : ''}</Timestamp>
                  </MetricRow>
                );
              })}
            </div>
          )}

          {/* Performance Section */}
          {(organizedData.performance.fps.length > 0 ||
            organizedData.performance.processing.length > 0 ||
            organizedData.performance.frames.length > 0) && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('performance')}
              >
                <span>📊 Performance</span>
                <SectionToggle collapsed={collapsedSections.performance}>▶</SectionToggle>
              </SectionHeader>

              {!collapsedSections.performance && (
                <>
                  {organizedData.performance.fps.length > 0 && (
                    <div>
                      <SubsectionHeader>FPS</SubsectionHeader>
                      <CameraGrid>
                        {organizedData.performance.fps
                          .sort((a, b) => a.group.localeCompare(b.group))
                          .map(item => {
                            const cameraName = item.group.replace('Camera ', '');
                            const secondsAgo = Math.round((now - item.ts) / 1000);
                            return (
                              <CameraCard key={item.group}>
                                <CameraName>{cameraName}</CameraName>
                                <CameraValue status={getCameraStatusColor(item.value, item.key)}>
                                  {String(item.value)}
                                </CameraValue>
                              </CameraCard>
                            );
                          })}
                      </CameraGrid>
                    </div>
                  )}

                  {organizedData.performance.processing.length > 0 && (
                    <div>
                      <SubsectionHeader>Processing Time (ms)</SubsectionHeader>
                      <CameraGrid>
                        {organizedData.performance.processing
                          .sort((a, b) => a.group.localeCompare(b.group))
                          .map(item => {
                            const cameraName = item.group.replace('Camera ', '');
                            return (
                              <CameraCard key={item.group}>
                                <CameraName>{cameraName}</CameraName>
                                <CameraValue>{String(item.value)}</CameraValue>
                              </CameraCard>
                            );
                          })}
                      </CameraGrid>
                    </div>
                  )}

                  {organizedData.performance.frames.length > 0 && (
                    <div>
                      <SubsectionHeader>Frame Count</SubsectionHeader>
                      <CameraGrid>
                        {organizedData.performance.frames
                          .sort((a, b) => a.group.localeCompare(b.group))
                          .map(item => {
                            const cameraName = item.group.replace('Camera ', '');
                            return (
                              <CameraCard key={item.group}>
                                <CameraName>{cameraName}</CameraName>
                                <CameraValue>{String(item.value)}</CameraValue>
                              </CameraCard>
                            );
                          })}
                      </CameraGrid>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ReID / StableID Section */}
          {organizedData.reid.length > 0 && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('details')}
              >
                <span>🧬 ReID / StableID</span>
                <SectionToggle collapsed={collapsedSections.details}>▶</SectionToggle>
              </SectionHeader>
              {!collapsedSections.details && organizedData.reid.map(item => (
                <MetricRow key={`reid-${item.key}`}>
                  <MetricLabel>{item.key}</MetricLabel>
                  <MetricValue>{String(item.value)}</MetricValue>
                  <Timestamp></Timestamp>
                </MetricRow>
              ))}
            </div>
          )}

          {/* Activity Section */}
          {organizedData.activity.length > 0 && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('activity')}
              >
                <span>🎯 Activity</span>
                <SectionToggle collapsed={collapsedSections.activity}>▶</SectionToggle>
              </SectionHeader>
              {!collapsedSections.activity && organizedData.activity.map(item => {
                const secondsAgo = Math.round((now - item.ts) / 1000);
                return (
                  <MetricRow key={`${item.group}-${item.key}`}>
                    <MetricLabel>{item.key}</MetricLabel>
                    <MetricValue>{String(item.value)}</MetricValue>
                    <Timestamp>{secondsAgo > 10 ? `${secondsAgo}s` : ''}</Timestamp>
                  </MetricRow>
                );
              })}
            </div>
          )}

          {/* Occupancy Section */}
          {organizedData.occupancy.length > 0 && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('occupancy')}
              >
                <span>🏢 Zone Occupancy</span>
                <SectionToggle collapsed={collapsedSections.occupancy}>▶</SectionToggle>
              </SectionHeader>
              {!collapsedSections.occupancy && (
                <OccupancyGrid>
                  {organizedData.occupancy.map(item => (
                    <OccupancyItem key={item.key}>
                      <ZoneName>{item.key}</ZoneName>
                      <ZoneCount>{String(item.value)}</ZoneCount>
                    </OccupancyItem>
                  ))}
                </OccupancyGrid>
              )}
            </div>
          )}

          {/* Details Section */}
          {organizedData.details.length > 0 && (
            <div>
              <SectionHeader
                collapsible
                onClick={() => toggleSection('details')}
              >
                <span>📋 Details</span>
                <SectionToggle collapsed={collapsedSections.details}>▶</SectionToggle>
              </SectionHeader>
              {!collapsedSections.details && organizedData.details.map(item => {
                const secondsAgo = Math.round((now - item.ts) / 1000);
                return (
                  <MetricRow key={`${item.group}-${item.key}`}>
                    <MetricLabel>{item.group} - {item.key}</MetricLabel>
                    <MetricValue>{String(item.value)}</MetricValue>
                    <Timestamp>{secondsAgo > 10 ? `${secondsAgo}s` : ''}</Timestamp>
                  </MetricRow>
                );
              })}
            </div>
          )}
        </div>
      </Drawer>
    </>
  );
};
