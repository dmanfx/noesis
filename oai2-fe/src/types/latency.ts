export type LatencyMetrics = {
  enabled?: boolean;
  window_sec?: number;
  count?: number;
  p50?: number | null;
  p95?: number | null;
  max?: number | null;
  last_sample_age_sec?: number | null;
  reason?: string;
};

