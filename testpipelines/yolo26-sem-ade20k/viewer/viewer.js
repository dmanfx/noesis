"use strict";

const SOURCE_URL = "/data/mapanything_phone_scans/20260802-162254-8bcc7dd7/fixed_camera_anchor_20260802/living-room_raw.jpg";
const LABELS_URL = "/models/yolo26_sem_ade20k/labels/ade20k_labels.txt";
const OUTPUT_ROOT = "/build/yolo26_sem_ade20k/captures/20260808_well_lit_anchor";
const MODEL_NAMES = { s: "Small", l: "Large" };
const PALETTE = [
  "#042aff", "#0bdbeb", "#f3f3f3", "#00dfb7", "#111f68",
  "#ff6fdd", "#ff444f", "#cced00", "#00f344", "#bd00ff",
  "#00b4ff", "#dd00ba", "#00ffff", "#26c000", "#01ffb3",
  "#7d24ff", "#7b0068", "#ff1b6c", "#fc6d2f", "#a2ff0b",
];

const canvas = document.querySelector("#semanticCanvas");
const context = canvas.getContext("2d", { alpha: false });
const stage = document.querySelector("#stage");
const tooltip = document.querySelector("#tooltip");
const tooltipSwatch = document.querySelector("#tooltipSwatch");
const tooltipLabel = document.querySelector("#tooltipLabel");
const tooltipMeta = document.querySelector("#tooltipMeta");
const readoutSwatch = document.querySelector("#readoutSwatch");
const readoutLabel = document.querySelector("#readoutLabel");
const status = document.querySelector("#status");
const loading = document.querySelector("#loading");
const modelSelect = document.querySelector("#modelSelect");
const opacitySlider = document.querySelector("#opacitySlider");
const opacityValue = document.querySelector("#opacityValue");
const overlayToggle = document.querySelector("#overlayToggle");
const legendGrid = document.querySelector("#legendGrid");
const legendSummary = document.querySelector("#legendSummary");
const clearFilter = document.querySelector("#clearFilter");

const state = {
  source: null,
  labels: [],
  classIds: null,
  mapWidth: 0,
  mapHeight: 0,
  model: modelSelect.value,
  opacity: Number(opacitySlider.value) / 100,
  overlay: overlayToggle.checked,
  summaries: [],
  selectedClassId: null,
};

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = url;
  });
}

function hsvToRgb(hue, saturation, value) {
  const chroma = value * saturation;
  const sector = hue / 60;
  const intermediate = chroma * (1 - Math.abs((sector % 2) - 1));
  let channels = [0, 0, 0];
  if (sector < 1) channels = [chroma, intermediate, 0];
  else if (sector < 2) channels = [intermediate, chroma, 0];
  else if (sector < 3) channels = [0, chroma, intermediate];
  else if (sector < 4) channels = [0, intermediate, chroma];
  else if (sector < 5) channels = [intermediate, 0, chroma];
  else channels = [chroma, 0, intermediate];
  const match = value - chroma;
  return channels.map((channel) => Math.round((channel + match) * 255));
}

function colorForClass(classId) {
  if (classId < PALETTE.length) return PALETTE[classId];
  const hue = (classId * 137.507764 + 29) % 360;
  const saturation = 0.66 + ((classId % 4) * 0.07);
  const value = Math.min(0.88 + ((classId % 3) * 0.04), 0.96);
  return `#${hsvToRgb(hue, saturation, value).map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

function rgbForClass(classId) {
  const hex = colorForClass(classId);
  return [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
  ];
}

function summarizeClassIds(classIds) {
  const counts = new Map();
  for (const classId of classIds) counts.set(classId, (counts.get(classId) || 0) + 1);
  return [...counts.entries()]
    .map(([classId, count]) => ({ classId, count, fraction: count / classIds.length }))
    .sort((left, right) => right.count - left.count || left.classId - right.classId);
}

function classMapUrl(model) {
  return `${OUTPUT_ROOT}/yolo26${model}/yolo26${model}_living_room_well_lit_class_map.png`;
}

async function decodeClassMap(model) {
  const mapImage = await loadImage(classMapUrl(model));
  const mapCanvas = document.createElement("canvas");
  mapCanvas.width = mapImage.naturalWidth;
  mapCanvas.height = mapImage.naturalHeight;
  const mapContext = mapCanvas.getContext("2d", { willReadFrequently: true });
  mapContext.drawImage(mapImage, 0, 0);
  const pixels = mapContext.getImageData(0, 0, mapCanvas.width, mapCanvas.height).data;
  const classIds = new Uint8Array(mapCanvas.width * mapCanvas.height);
  for (let index = 0; index < classIds.length; index += 1) {
    classIds[index] = pixels[index * 4];
  }
  return { classIds, width: mapCanvas.width, height: mapCanvas.height };
}

function render() {
  if (!state.source || !state.classIds) return;
  canvas.width = state.source.naturalWidth;
  canvas.height = state.source.naturalHeight;
  context.globalAlpha = 1;
  context.imageSmoothingEnabled = true;
  context.drawImage(state.source, 0, 0, canvas.width, canvas.height);
  if (state.selectedClassId === null && (!state.overlay || state.opacity <= 0)) return;

  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = state.mapWidth;
  overlayCanvas.height = state.mapHeight;
  const overlayContext = overlayCanvas.getContext("2d");
  const overlayImage = overlayContext.createImageData(state.mapWidth, state.mapHeight);
  for (let index = 0; index < state.classIds.length; index += 1) {
    const classId = state.classIds[index];
    const offset = index * 4;
    if (state.selectedClassId !== null && classId !== state.selectedClassId) {
      overlayImage.data[offset] = 0;
      overlayImage.data[offset + 1] = 0;
      overlayImage.data[offset + 2] = 0;
      overlayImage.data[offset + 3] = 218;
    } else if (state.overlay) {
      const [red, green, blue] = rgbForClass(classId);
      overlayImage.data[offset] = red;
      overlayImage.data[offset + 1] = green;
      overlayImage.data[offset + 2] = blue;
      overlayImage.data[offset + 3] = Math.round(state.opacity * 255);
    }
  }
  overlayContext.putImageData(overlayImage, 0, 0);
  context.imageSmoothingEnabled = false;
  context.drawImage(overlayCanvas, 0, 0, canvas.width, canvas.height);
}

function selectedSummary() {
  return state.summaries.find((item) => item.classId === state.selectedClassId) || null;
}

function updateReadout() {
  const selected = selectedSummary();
  if (selected) {
    readoutSwatch.style.background = colorForClass(selected.classId);
    readoutSwatch.style.opacity = "1";
    readoutLabel.textContent = `Filtered to ${state.labels[selected.classId]} · class ${selected.classId} · ${(selected.fraction * 100).toFixed(1)}% of map pixels`;
  } else {
    readoutSwatch.style.opacity = "0";
    readoutLabel.textContent = "Move over the image to inspect a pixel · select a segment below to isolate it";
  }
}

function renderLegend() {
  legendGrid.replaceChildren();
  legendSummary.textContent = `${state.summaries.length} of ${state.labels.length} ADE20K classes`;
  clearFilter.disabled = state.selectedClassId === null;
  for (const item of state.summaries) {
    const selected = state.selectedClassId === item.classId;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `legend-item${selected ? " active" : ""}`;
    button.setAttribute("aria-pressed", String(selected));
    button.title = selected ? "Click again to clear the filter" : `Isolate ${state.labels[item.classId]}`;

    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = colorForClass(item.classId);
    const label = document.createElement("span");
    label.className = "legend-label";
    const strong = document.createElement("strong");
    strong.textContent = state.labels[item.classId] || `class ${item.classId}`;
    const small = document.createElement("small");
    small.textContent = `class ${item.classId}`;
    label.append(strong, small);
    const share = document.createElement("span");
    share.className = "legend-share";
    share.textContent = `${(item.fraction * 100).toFixed(1)}%`;
    button.append(swatch, label, share);
    button.addEventListener("click", () => {
      state.selectedClassId = selected ? null : item.classId;
      tooltip.classList.remove("visible");
      render();
      renderLegend();
      updateReadout();
    });
    legendGrid.append(button);
  }
}

function canvasPixelAt(event) {
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const renderedWidth = canvas.width * scale;
  const renderedHeight = canvas.height * scale;
  const offsetX = (rect.width - renderedWidth) / 2;
  const offsetY = (rect.height - renderedHeight) / 2;
  const sourceX = (event.clientX - rect.left - offsetX) / scale;
  const sourceY = (event.clientY - rect.top - offsetY) / scale;
  if (sourceX < 0 || sourceY < 0 || sourceX >= canvas.width || sourceY >= canvas.height) {
    return null;
  }
  return { x: Math.floor(sourceX), y: Math.floor(sourceY) };
}

function inspectPixel(event) {
  if (!state.classIds) return;
  const pixel = canvasPixelAt(event);
  if (!pixel) {
    tooltip.classList.remove("visible");
    return;
  }
  const mapX = Math.min(state.mapWidth - 1, Math.floor(pixel.x * state.mapWidth / canvas.width));
  const mapY = Math.min(state.mapHeight - 1, Math.floor(pixel.y * state.mapHeight / canvas.height));
  const classId = state.classIds[mapY * state.mapWidth + mapX];
  const label = state.labels[classId] || `class ${classId}`;
  const color = colorForClass(classId);
  tooltipLabel.textContent = label;
  tooltipMeta.textContent = `class ${classId} · pixel ${pixel.x}, ${pixel.y}`;
  tooltipSwatch.style.background = color;
  readoutSwatch.style.background = color;
  readoutSwatch.style.opacity = "1";
  readoutLabel.textContent = `${MODEL_NAMES[state.model]} · ${label} · class ${classId} · source pixel (${pixel.x}, ${pixel.y})`;

  const stageRect = stage.getBoundingClientRect();
  const tooltipWidth = tooltip.offsetWidth || 160;
  const tooltipHeight = tooltip.offsetHeight || 54;
  const proposedX = event.clientX - stageRect.left + 14;
  const proposedY = event.clientY - stageRect.top + 14;
  tooltip.style.left = `${Math.min(stageRect.width - tooltipWidth - 8, proposedX)}px`;
  tooltip.style.top = `${Math.min(stageRect.height - tooltipHeight - 8, proposedY)}px`;
  tooltip.classList.add("visible");
}

async function loadModel(model) {
  state.model = model;
  state.selectedClassId = null;
  loading.textContent = `Loading ${MODEL_NAMES[model]} class map…`;
  loading.classList.remove("hidden");
  tooltip.classList.remove("visible");
  try {
    const decoded = await decodeClassMap(model);
    state.classIds = decoded.classIds;
    state.mapWidth = decoded.width;
    state.mapHeight = decoded.height;
    state.summaries = summarizeClassIds(decoded.classIds);
    render();
    renderLegend();
    updateReadout();
    status.textContent = `${MODEL_NAMES[model]} · ${decoded.width}×${decoded.height} class map · ${state.summaries.length} classes present`;
  } catch (error) {
    status.textContent = error.message;
    loading.textContent = error.message;
    throw error;
  } finally {
    if (state.classIds) loading.classList.add("hidden");
  }
}

async function initialize() {
  try {
    const [source, labelsResponse] = await Promise.all([loadImage(SOURCE_URL), fetch(LABELS_URL)]);
    if (!labelsResponse.ok) throw new Error(`Could not load ADE20K labels (${labelsResponse.status})`);
    state.source = source;
    state.labels = (await labelsResponse.text()).split(/\r?\n/).map((label) => label.trim()).filter(Boolean);
    if (state.labels.length !== 150) throw new Error(`Expected 150 labels, found ${state.labels.length}`);
    await loadModel(state.model);
  } catch (error) {
    status.textContent = error.message;
    loading.textContent = error.message;
  }
}

modelSelect.addEventListener("change", () => loadModel(modelSelect.value));
opacitySlider.addEventListener("input", () => {
  state.opacity = Number(opacitySlider.value) / 100;
  opacityValue.value = `${opacitySlider.value}%`;
  render();
});
overlayToggle.addEventListener("change", () => {
  state.overlay = overlayToggle.checked;
  render();
});
canvas.addEventListener("mousemove", inspectPixel);
canvas.addEventListener("mouseleave", () => {
  tooltip.classList.remove("visible");
  updateReadout();
});
clearFilter.addEventListener("click", () => {
  state.selectedClassId = null;
  tooltip.classList.remove("visible");
  render();
  renderLegend();
  updateReadout();
});
window.addEventListener("resize", () => tooltip.classList.remove("visible"));

initialize();
