export type CameraKey = 'living-room' | 'kitchen' | 'family-room';

export function detectCameraKey(rawId: string): CameraKey | null {
  const id = (rawId || '').toLowerCase().trim();
  if (id === 'rtsp_0' || id.includes('living') || id.includes('room1') || id === '1') return 'living-room';
  if (id === 'rtsp_1' || id.includes('kitchen') || id.includes('room2') || id === '2') return 'kitchen';
  if (id === 'rtsp_2' || id.includes('family') || id.includes('room3') || id === '3') return 'family-room';
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
  return 2;
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
  return 'Family Room';
}
