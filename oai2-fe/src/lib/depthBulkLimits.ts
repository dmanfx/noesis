// Mirrored runtime constants. tests/depthBulkLimits.test.ts binds these values
// to noesis_core/depth_bulk_limits.json so frontend and backend cannot drift.
export const DEPTH_BULK_MAX_PIXELS = 8_388_608;
export const DEPTH_BULK_MAX_COMPONENT_BYTES = 64 * 1024 * 1024;
export const DEPTH_BULK_MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024;

export const DEPTH_BULK_REQUEST_TIMEOUT_MS = 60_000;
export const DEPTH_BULK_MAX_ACTIVE_REQUESTS = 3;
export const DEPTH_BULK_MAX_ACTIVE_ALLOCATION_BYTES = 256 * 1024 * 1024;
