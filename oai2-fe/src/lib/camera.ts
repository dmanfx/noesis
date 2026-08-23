export type CameraKey = string; // Dynamic discovery + backward compat for the original three known rooms

export function detectCameraKey(rawId: string): CameraKey | null {
  const id = (rawId || '').toLowerCase().trim();

  // Strong / exact aliases for the original three rooms only.
  // We deliberately avoid broad .includes() on common words like "kitchen", "family", "living"
  // to prevent dynamic camera names (e.g. "kitchen-2", "family-room-side") from being
  // incorrectly collapsed (per Codex medium finding).
  const LIVING_ALIASES = ['living-room', 'livingroom', 'living_room', 'rtsp_0', 'room1', '1'];
  const KITCHEN_ALIASES = ['kitchen', 'kitchen-room', 'kitchen_room', 'rtsp_1', 'room2', '2'];
  const FAMILY_ALIASES = ['family-room', 'familyroom', 'family_room', 'rtsp_2', 'room3', '3'];

  if (LIVING_ALIASES.includes(id)) return 'living-room';
  if (KITCHEN_ALIASES.includes(id)) return 'kitchen';
  if (FAMILY_ALIASES.includes(id)) return 'family-room';

  // Very conservative partials only for the classic old forms (not general words)
  if (id === 'liv' || id.startsWith('liv-') || id.startsWith('liv_')) return 'living-room';
  if (id === 'kit' || id.startsWith('kit-') || id.startsWith('kit_')) return 'kitchen';
  if (id === 'fam' || id.startsWith('fam-') || id.startsWith('fam_')) return 'family-room';

  return null;
}

export const cameraOrder: CameraKey[] = ['living-room', 'kitchen', 'family-room'];

export const COLOR_NS_STRIDE = 2 ** 32;

export function colorForTrack(id: number): string {
  const hue = (id * 47) % 360;
  return `hsl(${hue}, 80%, 60%)`;
}

export function cameraIndex(cam: CameraKey): number {
  if (cam === 'living-room') return 0;
  if (cam === 'kitchen') return 1;
  if (cam === 'family-room') return 2;
  // Dynamic cameras: stable hash-based index (good enough for colors/legends)
  let hash = 0;
  for (let i = 0; i < cam.length; i++) {
    hash = (hash * 31 + cam.charCodeAt(i)) | 0;
  }
  return Math.abs(hash) % 16 + 3; // start after the original three
}

export function colorIdForPerson(_cam: CameraKey, stableId?: number | null): number {
  const stable = typeof stableId === 'number' && Number.isFinite(stableId) ? stableId : null;
  if (stable !== null && stable > 0) return stable;
  return 0;
}

export function identityKeyForPerson(_cam: CameraKey, stableId?: number | null): string {
  const stable = typeof stableId === 'number' && Number.isFinite(stableId) ? stableId : null;
  if (stable !== null && stable > 0) return `s:${stable}`;
  return 's:0';
}

export function cameraLabel(key: CameraKey): string {
  if (key === 'living-room') return 'Living Room';
  if (key === 'kitchen') return 'Kitchen';
  if (key === 'family-room') return 'Family Room';
  // Fallback for dynamically discovered cameras
  return key.replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

const CALIBRATION_TABLE_KEYS = new Set([
  'k',
  'e',
  'pose',
  'pose confidence',
  'pose_confidence',
  'intrinsics',
  'extrinsics',
]);

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function isCalibrationTableKey(key: string): boolean {
  return CALIBRATION_TABLE_KEYS.has(key.toLowerCase().trim());
}

function addCameraKey(discovered: Set<CameraKey>, rawId: unknown): void {
  const id = String(rawId || '').trim();
  if (!id) return;
  const key = detectCameraKey(id) || id.toLowerCase();
  if (key) discovered.add(key);
}

/**
 * Discovery helper for Item 4 (config + discovery approach).
 * Collects camera identifiers from the first successful feeds:
 * - calibration-bundle
 * - stats.pipeline.mosaic_layout.sources
 * - floorplan responses
 * The original three are always seeded for backward compat.
 */
export function discoverCamerasFromPayloads(payloads: any[]): CameraKey[] {
  const discovered = new Set<CameraKey>(['living-room', 'kitchen', 'family-room']); // seed for compat

  for (const p of payloads) {
    if (!p) continue;

    // From mosaic layout sources
    const layout = p.pipeline?.mosaic_layout || p.mosaic_layout;
    if (layout && Array.isArray(layout.sources)) {
      for (const src of layout.sources) {
        const id = src?.camera_id || src?.cameraId || src?.source_id;
        if (id) addCameraKey(discovered, id);
      }
    }

    // From calibration bundle
    if (p.data?.cameras || p.cameras) {
      const cams = p.data?.cameras || p.cameras;
      for (const [key, value] of Object.entries(cams || {})) {
        if (isCalibrationTableKey(key)) {
          if (isObjectRecord(value)) {
            Object.keys(value).forEach((cameraId) => addCameraKey(discovered, cameraId));
          }
          continue;
        }
        addCameraKey(discovered, key);
      }
    }

    // From floorplan or bev meta
    const cam = p.camera || p.cameraId || p.camId || p.camera_id || p.cam_id;
    if (cam) addCameraKey(discovered, cam);
  }

  return Array.from(discovered);
}
