import {
  MouseEvent as ReactMouseEvent,
  PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ADE20K_LABELS,
  semanticColor,
  semanticColorRgb,
  summarizeClassIds,
  toggleSemanticClass,
  type SemanticClassSummary,
} from '../lib/semanticSegmentation.js';
import '../styles/semantic-seg.css';

type ModelSize = 's' | 'l';
type CameraId = 'living-room' | 'kitchen' | 'family-room';

type DecodedClassMap = {
  classIds: Uint8Array;
  width: number;
  height: number;
  summaries: SemanticClassSummary[];
};

type HoverReadout = {
  classId: number;
  sourceX: number;
  sourceY: number;
  left: number;
  top: number;
};

type RuntimeCameraCapture = {
  id: CameraId;
  label: string;
  source_id: number;
  width: number;
  height: number;
  raw_url: string;
  class_map_url: string;
  masked_url: string;
};

type RuntimeCapture = {
  contract: 'noesis.semantic_seg.capture';
  contract_version: 1;
  capture_id: string;
  model: ModelSize;
  model_name: string;
  captured_at: string;
  cameras: RuntimeCameraCapture[];
};

const MODEL_OPTIONS: Array<{ value: ModelSize; label: string }> = [
  { value: 's', label: 'Small' },
  { value: 'l', label: 'Large' },
];
const CAMERA_OPTIONS: Array<{ value: CameraId; label: string; stem: string; evidence: string }> = [
  { value: 'living-room', label: 'Living Room', stem: 'source_0_living_room', evidence: 'saved well-lit anchor' },
  { value: 'kitchen', label: 'Kitchen', stem: 'source_1_kitchen', evidence: 'fresh three-camera capture' },
  { value: 'family-room', label: 'Family Room', stem: 'source_2_family_room', evidence: 'fresh three-camera capture' },
];
const MODEL_LABELS: Record<ModelSize, string> = Object.fromEntries(
  MODEL_OPTIONS.map((option) => [option.value, option.label]),
) as Record<ModelSize, string>;
const CAMERA_LABELS: Record<CameraId, string> = Object.fromEntries(
  CAMERA_OPTIONS.map((option) => [option.value, option.label]),
) as Record<CameraId, string>;
const ASSET_ROOT = `${import.meta.env.BASE_URL}semantic-seg`;
const imagePromises = new Map<string, Promise<HTMLImageElement>>();
const classMapPromises = new Map<string, Promise<DecodedClassMap>>();

function loadImage(url: string): Promise<HTMLImageElement> {
  const existing = imagePromises.get(url);
  if (existing) return existing;
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = url;
  });
  imagePromises.set(url, promise);
  promise.catch(() => imagePromises.delete(url));
  return promise;
}

function cameraStem(camera: CameraId): string {
  return CAMERA_OPTIONS.find((option) => option.value === camera)?.stem ?? CAMERA_OPTIONS[0].stem;
}

function cameraEvidence(camera: CameraId): string {
  return CAMERA_OPTIONS.find((option) => option.value === camera)?.evidence ?? CAMERA_OPTIONS[0].evidence;
}

function staticSourceUrl(model: ModelSize, camera: CameraId): string {
  return `${ASSET_ROOT}/three-camera/yolo26${model}_${cameraStem(camera)}_raw.jpg`;
}

function staticClassMapUrl(model: ModelSize, camera: CameraId): string {
  return `${ASSET_ROOT}/three-camera/yolo26${model}_${cameraStem(camera)}_class_map.png`;
}

function loadClassMap(url: string): Promise<DecodedClassMap> {
  const existing = classMapPromises.get(url);
  if (existing) return existing;
  const promise = loadImage(url).then((image) => {
    const mapCanvas = document.createElement('canvas');
    mapCanvas.width = image.naturalWidth;
    mapCanvas.height = image.naturalHeight;
    const mapContext = mapCanvas.getContext('2d', { willReadFrequently: true });
    if (!mapContext) throw new Error('Canvas class-map decoding is not available.');
    mapContext.drawImage(image, 0, 0);
    const pixels = mapContext.getImageData(0, 0, mapCanvas.width, mapCanvas.height).data;
    const classIds = new Uint8Array(mapCanvas.width * mapCanvas.height);
    for (let index = 0; index < classIds.length; index += 1) {
      classIds[index] = pixels[index * 4];
    }
    return {
      classIds,
      width: mapCanvas.width,
      height: mapCanvas.height,
      summaries: summarizeClassIds(classIds),
    };
  });
  classMapPromises.set(url, promise);
  promise.catch(() => classMapPromises.delete(url));
  return promise;
}

function runtimeCamera(capture: RuntimeCapture | undefined, camera: CameraId): RuntimeCameraCapture | undefined {
  return capture?.cameras.find((item) => item.id === camera);
}

function isRuntimeCapture(value: unknown, expectedModel: ModelSize): value is RuntimeCapture {
  const payload = value as Partial<RuntimeCapture> | null;
  return payload?.contract === 'noesis.semantic_seg.capture'
    && payload.contract_version === 1
    && payload.model === expectedModel
    && typeof payload.capture_id === 'string'
    && typeof payload.captured_at === 'string'
    && Array.isArray(payload.cameras)
    && CAMERA_OPTIONS.every((option) => payload.cameras?.some((item) => (
      item?.id === option.value
      && typeof item.raw_url === 'string'
      && typeof item.class_map_url === 'string'
    )));
}

function SemanticSegView() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [camera, setCamera] = useState<CameraId>('living-room');
  const [model, setModel] = useState<ModelSize>('l');
  const [opacity, setOpacity] = useState(0.55);
  const [showOverlay, setShowOverlay] = useState(true);
  const [sourceImage, setSourceImage] = useState<HTMLImageElement | null>(null);
  const [classMap, setClassMap] = useState<DecodedClassMap | null>(null);
  const [selectedClassIds, setSelectedClassIds] = useState<number[]>([]);
  const [hover, setHover] = useState<HoverReadout | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runtimeCaptures, setRuntimeCaptures] = useState<Partial<Record<ModelSize, RuntimeCapture>>>({});
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  const activeCapture = runtimeCaptures[model];
  const activeRuntimeCamera = runtimeCamera(activeCapture, camera);
  const activeSourceUrl = activeRuntimeCamera?.raw_url ?? staticSourceUrl(model, camera);
  const activeClassMapUrl = activeRuntimeCamera?.class_map_url ?? staticClassMapUrl(model, camera);

  useEffect(() => {
    let current = true;
    setLoading(true);
    setError(null);
    setHover(null);
    Promise.all([loadImage(activeSourceUrl), loadClassMap(activeClassMapUrl)])
      .then(([source, decoded]) => {
        if (!current) return;
        setSourceImage(source);
        setClassMap(decoded);
      })
      .catch((reason: unknown) => {
        if (!current) return;
        setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => { current = false; };
  }, [activeClassMapUrl, activeSourceUrl, camera, model]);

  const refreshCapture = async () => {
    if (refreshing) return;
    setRefreshing(true);
    setRefreshError(null);
    try {
      const response = await fetch('/api/diagnostics/semantic-seg/captures', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Idempotency-Key': `semseg-${model}-${crypto.randomUUID()}`,
        },
        body: JSON.stringify({ model }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(payload?.message || payload?.detail || `Semantic capture failed with HTTP ${response.status}`);
      }
      if (!isRuntimeCapture(payload, model)) {
        throw new Error('The semantic capture response was incomplete or invalid.');
      }
      setRuntimeCaptures((current) => ({ ...current, [model]: payload }));
      setSelectedClassIds([]);
      setHover(null);
    } catch (reason: unknown) {
      setRefreshError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sourceImage || !classMap) return;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;
    canvas.width = sourceImage.naturalWidth;
    canvas.height = sourceImage.naturalHeight;
    context.globalAlpha = 1;
    context.imageSmoothingEnabled = true;
    context.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

    if (selectedClassIds.length === 0 && (!showOverlay || opacity <= 0)) return;
    const selectedClassIdSet = new Set(selectedClassIds);
    const overlayCanvas = document.createElement('canvas');
    overlayCanvas.width = classMap.width;
    overlayCanvas.height = classMap.height;
    const overlayContext = overlayCanvas.getContext('2d');
    if (!overlayContext) return;
    const overlay = overlayContext.createImageData(classMap.width, classMap.height);

    for (let index = 0; index < classMap.classIds.length; index += 1) {
      const classId = classMap.classIds[index];
      const offset = index * 4;
      if (selectedClassIdSet.size > 0 && !selectedClassIdSet.has(classId)) {
        overlay.data[offset] = 0;
        overlay.data[offset + 1] = 0;
        overlay.data[offset + 2] = 0;
        overlay.data[offset + 3] = 218;
        continue;
      }
      if (showOverlay) {
        const [red, green, blue] = semanticColorRgb(classId);
        overlay.data[offset] = red;
        overlay.data[offset + 1] = green;
        overlay.data[offset + 2] = blue;
        overlay.data[offset + 3] = Math.round(opacity * 255);
      }
    }
    overlayContext.putImageData(overlay, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(overlayCanvas, 0, 0, canvas.width, canvas.height);
  }, [classMap, opacity, selectedClassIds, showOverlay, sourceImage]);

  const selectedClassIdSet = useMemo(() => new Set(selectedClassIds), [selectedClassIds]);
  const selectedSummaries = useMemo(
    () => classMap?.summaries.filter((item) => selectedClassIdSet.has(item.classId)) ?? [],
    [classMap, selectedClassIdSet],
  );
  const selectedFraction = selectedSummaries.reduce((total, item) => total + item.fraction, 0);
  const selectedLabels = selectedSummaries.map(
    (item) => ADE20K_LABELS[item.classId] ?? `class ${item.classId}`,
  );
  const selectedLabelSummary = selectedLabels.length <= 3
    ? selectedLabels.join(', ')
    : `${selectedLabels.slice(0, 3).join(', ')} +${selectedLabels.length - 3}`;

  const readPixel = (event: { clientX: number; clientY: number }): HoverReadout | null => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage || !classMap) return null;
    const canvasRect = canvas.getBoundingClientRect();
    const sourceX = Math.min(
      canvas.width - 1,
      Math.max(0, Math.floor(((event.clientX - canvasRect.left) / canvasRect.width) * canvas.width)),
    );
    const sourceY = Math.min(
      canvas.height - 1,
      Math.max(0, Math.floor(((event.clientY - canvasRect.top) / canvasRect.height) * canvas.height)),
    );
    const mapX = Math.min(classMap.width - 1, Math.floor((sourceX * classMap.width) / canvas.width));
    const mapY = Math.min(classMap.height - 1, Math.floor((sourceY * classMap.height) / canvas.height));
    const classId = classMap.classIds[(mapY * classMap.width) + mapX];
    const stageRect = stage.getBoundingClientRect();
    return {
      classId,
      sourceX,
      sourceY,
      left: Math.min(stageRect.width - 188, Math.max(8, event.clientX - stageRect.left + 14)),
      top: Math.min(stageRect.height - 62, Math.max(8, event.clientY - stageRect.top + 14)),
    };
  };

  const inspectPixel = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    setHover(readPixel(event));
  };

  const togglePixel = (event: ReactMouseEvent<HTMLCanvasElement>) => {
    const inspected = readPixel(event);
    if (!inspected) return;
    setSelectedClassIds((current) => toggleSemanticClass(current, inspected.classId));
    setHover(inspected);
  };

  const selectClass = (classId: number) => {
    setSelectedClassIds((current) => toggleSemanticClass(current, classId));
    setHover(null);
  };

  return (
    <section className="semseg-view" aria-label="Semantic segmentation inspector">
      <div className="semseg-view__heading">
        <div>
          <span>
            {CAMERA_LABELS[camera]} · {activeCapture
              ? `live capture ${new Date(activeCapture.captured_at).toLocaleTimeString()}`
              : cameraEvidence(camera)}
          </span>
          <strong>YOLO26 semantic segmentation · ADE20K</strong>
        </div>
        <span className="semseg-view__display-only">
          {refreshing ? `Capturing ${MODEL_LABELS[model]} on 3 cameras…` : 'Manual batch-3 capture'}
        </span>
      </div>

      <div className="semseg-view__controls">
        <label>
          <span>Camera</span>
          <select
            aria-label="Semantic segmentation camera"
            value={camera}
            disabled={refreshing}
            onChange={(event) => {
              setCamera(event.target.value as CameraId);
              setSelectedClassIds([]);
            }}
          >
            {CAMERA_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Model</span>
          <select
            aria-label="Semantic segmentation model"
            value={model}
            disabled={refreshing}
            onChange={(event) => {
              setModel(event.target.value as ModelSize);
              setSelectedClassIds([]);
            }}
          >
            {MODEL_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label className="semseg-view__opacity">
          <span>Overlay <output>{Math.round(opacity * 100)}%</output></span>
          <input
            type="range"
            min="0"
            max="100"
            value={Math.round(opacity * 100)}
            onChange={(event) => setOpacity(Number(event.target.value) / 100)}
          />
        </label>
        <label className="semseg-view__toggle">
          <input
            type="checkbox"
            checked={showOverlay}
            onChange={(event) => setShowOverlay(event.target.checked)}
          />
          <span>Show mask</span>
        </label>
        <button
          type="button"
          className="semseg-view__refresh"
          disabled={refreshing}
          onClick={refreshCapture}
        >
          {refreshing ? 'Capturing…' : 'Refresh capture'}
        </button>
      </div>

      {refreshError && <div className="semseg-view__capture-error" role="alert">{refreshError}</div>}

      <div
        className="semseg-view__stage"
        ref={stageRef}
        style={{ aspectRatio: sourceImage ? `${sourceImage.naturalWidth} / ${sourceImage.naturalHeight}` : '16 / 9' }}
      >
        <canvas
          ref={canvasRef}
          aria-label={`${MODEL_LABELS[model]} semantic segmentation of ${CAMERA_LABELS[camera]}. Click a pixel to add or remove its class from the filter.`}
          onClick={togglePixel}
          onPointerMove={inspectPixel}
          onPointerLeave={() => setHover(null)}
        />
        {hover && (
          <div className="semseg-view__tooltip" style={{ left: hover.left, top: hover.top }} role="status">
            <span style={{ background: semanticColor(hover.classId) }} />
            <div>
              <strong>{ADE20K_LABELS[hover.classId] ?? `class ${hover.classId}`}</strong>
              <small>class {hover.classId} · pixel {hover.sourceX}, {hover.sourceY}</small>
            </div>
          </div>
        )}
        {(loading || error) && (
          <div className={`semseg-view__loading${error ? ' semseg-view__loading--error' : ''}`}>
            {error ?? `Loading ${CAMERA_LABELS[camera]} · ${MODEL_LABELS[model]} class map…`}
          </div>
        )}
      </div>

      <div className="semseg-view__readout">
        <span>
          {selectedSummaries.length > 0
            ? `Filtered to ${selectedSummaries.length} segment${selectedSummaries.length === 1 ? '' : 's'}: ${selectedLabelSummary} · ${(selectedFraction * 100).toFixed(1)}% of map pixels`
            : 'Move over the image to inspect a pixel. Click the image or use the legend to toggle one or more classes.'}
        </span>
        {classMap && <span>{CAMERA_LABELS[camera]} · {MODEL_LABELS[model]} · {classMap.width}×{classMap.height} class map</span>}
      </div>

      <section className="semseg-view__legend" aria-label="Classes present in this segmentation">
        <div className="semseg-view__legend-heading">
          <div>
            <strong>Segments present</strong>
            <span>{classMap?.summaries.length ?? 0} of {ADE20K_LABELS.length} ADE20K classes</span>
          </div>
          <button
            type="button"
            disabled={selectedClassIds.length === 0}
            onClick={() => setSelectedClassIds([])}
          >
            Clear filter
          </button>
        </div>
        <div className="semseg-view__legend-grid">
          {classMap?.summaries.map((item) => {
            const selected = selectedClassIdSet.has(item.classId);
            return (
              <button
                type="button"
                key={item.classId}
                className={selected ? 'active' : ''}
                aria-pressed={selected}
                onClick={() => selectClass(item.classId)}
                title={selected
                  ? `Remove ${ADE20K_LABELS[item.classId]} from the filter`
                  : `Add ${ADE20K_LABELS[item.classId]} to the filter`}
              >
                <span className="semseg-view__legend-swatch" style={{ background: semanticColor(item.classId) }} />
                <span className="semseg-view__legend-label">
                  <strong>{ADE20K_LABELS[item.classId] ?? `class ${item.classId}`}</strong>
                  <small>class {item.classId}</small>
                </span>
                <span className="semseg-view__legend-share">{(item.fraction * 100).toFixed(1)}%</span>
              </button>
            );
          })}
        </div>
      </section>
    </section>
  );
}

export default SemanticSegView;
