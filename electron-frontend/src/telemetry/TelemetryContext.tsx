import React, {createContext, useContext, useState} from 'react';

export type TelemetryEntry = {
  group: string;
  key: string;
  value: string | number;
  ts: number; // epoch ms
};

interface TelemetryContextValue {
  entries: TelemetryEntry[];
  publish: (entry: TelemetryEntry) => void;
}

const TelemetryContext = createContext<TelemetryContextValue>({
  entries: [],
  publish: () => {},
});

export const TelemetryProvider: React.FC<{children: React.ReactNode}> = ({children}) => {
  const [entries, setEntries] = useState<TelemetryEntry[]>([]);

  const publish = (entry: TelemetryEntry) => {
    setEntries(prev => {
      // replace existing entry with same group+key
      const filtered = prev.filter(e => !(e.group === entry.group && e.key === entry.key));
      return [...filtered, entry];
    });
  };

  return (
    <TelemetryContext.Provider value={{entries, publish}}>
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetry = () => useContext(TelemetryContext);
