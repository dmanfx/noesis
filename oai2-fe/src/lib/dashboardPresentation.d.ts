export type BevLayer = 'structure' | 'texture' | 'height' | 'normals';
export type ScientificPalette = 'turbo' | 'viridis' | 'gray';
export type CameraDisplayMode = 'video' | 'depth' | 'normals';
export type AppearanceTheme = 'dark' | 'light' | 'contrast';
export type PresentationSettings = {
  theme: AppearanceTheme; bevLayer: BevLayer; palette: ScientificPalette; opacity: number;
  people: boolean; trails: boolean; cameras: boolean; labels: boolean;
  defaultCameraMode: CameraDisplayMode; cameraModes: Record<string, CameraDisplayMode>;
  comparison: number;
};
export type SavedPresentationPreset = { id: string; name: string; settings: PresentationSettings };
export const PRESENTATION_STORAGE_KEY: string;
export function defaultPresentationSettings(): PresentationSettings;
export function sanitizePresentationSettings(value: unknown): PresentationSettings;
export function sanitizeSavedPresets(value: unknown): SavedPresentationPreset[];
export function applyPresentationPreset(current: PresentationSettings, preset: 'monitor' | 'geometry' | 'surface'): PresentationSettings;
export function cameraDisplayMode(settings: PresentationSettings, camera: string): CameraDisplayMode;
export function allCameraDisplayMode(settings: PresentationSettings, cameras: readonly string[]): CameraDisplayMode | 'mixed';
