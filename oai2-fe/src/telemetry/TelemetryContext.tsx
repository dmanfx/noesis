import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

export type TelemetryEntry = { group: string; key: string; value: string | number; ts: number };

interface TelemetryValue {
  entries: TelemetryEntry[];
  publish: (e: TelemetryEntry) => void;
  clear: () => void;
}

const Ctx = createContext<TelemetryValue>({ entries: [], publish: () => {}, clear: () => {} });

export const TelemetryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [entries, setEntries] = useState<TelemetryEntry[]>([]);

  const publish = (e: TelemetryEntry) => {
    setEntries(prev => {
      const filtered = prev.filter(x => !(x.group === e.group && x.key === e.key));
      return [...filtered, e];
    });
  };
  const clear = () => setEntries([]);

  useEffect(() => {
    (window as any).publishTelemetry = publish;
    return () => { delete (window as any).publishTelemetry; };
  }, []);

  const value = useMemo(() => ({ entries, publish, clear }), [entries]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
};

export const useTelemetry = () => useContext(Ctx);

