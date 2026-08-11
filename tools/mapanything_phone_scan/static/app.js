import * as THREE from "three";
import { OrbitControls } from "/three/examples/controls/OrbitControls.js";
import { GLTFLoader } from "/three/examples/loaders/GLTFLoader.js";

const healthPill = document.querySelector("#health-pill");
const scanList = document.querySelector("#scan-list");
const scanDetail = document.querySelector("#scan-detail");
const scanName = document.querySelector("#scan-name");
const cameraInput = document.querySelector("#camera-video");
const existingInput = document.querySelector("#existing-video");
const uploadPanel = document.querySelector("#upload-panel");
const uploadLabel = document.querySelector("#upload-label");
const uploadPercent = document.querySelector("#upload-percent");
const uploadBar = document.querySelector("#upload-bar");
const refreshButton = document.querySelector("#refresh-button");
const newWalkButton = document.querySelector("#new-walk-button");
const captureCard = document.querySelector(".capture-card");
const workspace = document.querySelector(".workspace");
const toast = document.querySelector("#toast");

const NEW_WALK_MODE_KEY = "phoneScanNewWalkMode";
const LONG_PRESS_MS = 650;

let scans = [];
let newWalkMode = localStorage.getItem(NEW_WALK_MODE_KEY) === "1";
let selectedId = newWalkMode ? null : localStorage.getItem("phoneScanSelected") || null;
let lastDetailFingerprint = "";
let outputPage = 0;
let currentViewer = null;
let refreshing = false;
let toastTimer = null;
let uploadInProgress = false;
let renamingScanId = null;
let suppressScanClicksUntil = 0;

const statusLabels = {
  uploading: "Uploading",
  processing_frames: "Preparing frames",
  ready: "Ready for inference",
  frame_failed: "Frame preparation failed",
  ma_queued: "MA queued",
  ma_running: "MapAnything running",
  ma_failed: "MapAnything failed",
  da3_queued: "DA3 queued",
  da3_running: "DA3 running",
  da3_failed: "DA3 failed",
  complete: "Complete",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const power = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** power).toFixed(power === 0 ? 0 : 1)} ${units[power]}`;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown time";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  const mins = Math.floor(value / 60);
  const secs = Math.round(value % 60);
  return mins ? `${mins}m ${secs}s` : `${secs}s`;
}

function toastMessage(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3600);
}

function setScanDetailVisible(visible) {
  scanDetail.classList.toggle("hidden", !visible);
  if (visible) scanDetail.removeAttribute("aria-hidden");
  else scanDetail.setAttribute("aria-hidden", "true");
}

function syncCaptureModeUi() {
  newWalkButton.classList.toggle("active", newWalkMode);
  newWalkButton.setAttribute("aria-pressed", newWalkMode ? "true" : "false");
  captureCard.classList.toggle("new-walk-mode", newWalkMode);
  workspace.classList.toggle("new-walk-mode", newWalkMode);
  if (newWalkMode) {
    scanDetail.replaceChildren();
    setScanDetailVisible(false);
  }
}

function rememberSelectedScan(scanId) {
  selectedId = scanId;
  newWalkMode = false;
  localStorage.removeItem(NEW_WALK_MODE_KEY);
  if (selectedId) localStorage.setItem("phoneScanSelected", selectedId);
  else localStorage.removeItem("phoneScanSelected");
  syncCaptureModeUi();
}

function openScan(scanId) {
  rememberSelectedScan(scanId);
  outputPage = 0;
  lastDetailFingerprint = "";
  renderScanList();
  renderSelected();
}

function startNewWalk() {
  if (uploadInProgress) {
    toastMessage("The current walk is still uploading. It will open as soon as it is safely saved.");
    return;
  }
  newWalkMode = true;
  selectedId = null;
  localStorage.setItem(NEW_WALK_MODE_KEY, "1");
  localStorage.removeItem("phoneScanSelected");
  scanName.value = "Phone room walk";
  cameraInput.value = "";
  existingInput.value = "";
  uploadPanel.classList.add("hidden");
  outputPage = 0;
  lastDetailFingerprint = "";
  disposeViewer();
  syncCaptureModeUi();
  renderScanList();
  renderSelected();
  captureCard.scrollIntoView({ behavior: "smooth", block: "start" });
  toastMessage("Ready for a new walk. Your earlier walks remain auto-saved.");
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch (_) {
      // The HTTP status remains the useful error.
    }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

async function renameScan(scanId) {
  if (renamingScanId) return;
  const scan = scans.find((item) => item.id === scanId);
  if (!scan) return;
  renamingScanId = scanId;
  try {
    const proposed = window.prompt("Rename this saved walk", scan.name);
    if (proposed === null) return;
    const name = proposed.trim().replace(/\s+/g, " ");
    if (!name) {
      toastMessage("A walk name cannot be empty");
      return;
    }
    if (name.length > 80) {
      toastMessage("A walk name can contain at most 80 characters");
      return;
    }
    if (name === scan.name) return;
    const updated = await jsonFetch(
      `/api/scans/${encodeURIComponent(scanId)}?name=${encodeURIComponent(name)}`,
      { method: "PATCH" },
    );
    const index = scans.findIndex((item) => item.id === scanId);
    if (index >= 0) scans[index] = updated;
    lastDetailFingerprint = "";
    renderScanList();
    renderSelected();
    toastMessage(`Walk renamed to “${name}” and saved`);
  } catch (error) {
    toastMessage(`Rename failed: ${error.message}`);
  } finally {
    renamingScanId = null;
  }
}

function wireScanCard(button) {
  const scanId = button.dataset.scanId;
  let holdTimer = null;
  let holdTriggered = false;
  let pointerStartX = 0;
  let pointerStartY = 0;

  const cancelHold = () => {
    if (holdTimer !== null) clearTimeout(holdTimer);
    holdTimer = null;
  };

  button.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    pointerStartX = event.clientX;
    pointerStartY = event.clientY;
    holdTriggered = false;
    cancelHold();
    holdTimer = setTimeout(() => {
      holdTimer = null;
      holdTriggered = true;
      suppressScanClicksUntil = performance.now() + 1200;
      void renameScan(scanId);
    }, LONG_PRESS_MS);
  });
  button.addEventListener("pointermove", (event) => {
    if (Math.hypot(event.clientX - pointerStartX, event.clientY - pointerStartY) > 12) {
      cancelHold();
    }
  });
  button.addEventListener("pointerup", cancelHold);
  button.addEventListener("pointercancel", cancelHold);
  button.addEventListener("pointerleave", cancelHold);
  button.addEventListener("click", (event) => {
    if (holdTriggered || performance.now() < suppressScanClicksUntil) {
      event.preventDefault();
      event.stopPropagation();
      holdTriggered = false;
      return;
    }
    openScan(scanId);
  });
  button.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    cancelHold();
    if (performance.now() < suppressScanClicksUntil) return;
    suppressScanClicksUntil = performance.now() + 1200;
    void renameScan(scanId);
  });
  button.addEventListener("keydown", (event) => {
    if (event.key !== "F2") return;
    event.preventDefault();
    void renameScan(scanId);
  });
}

async function refreshHealth() {
  try {
    const health = await jsonFetch("/api/health");
    healthPill.textContent = `${health.device} · ${health.max_frames} views max`;
    healthPill.className = "health-pill online";
  } catch (error) {
    healthPill.textContent = "Tool offline";
    healthPill.className = "health-pill offline";
  }
}

function renderScanList() {
  if (!scans.length) {
    scanList.innerHTML = '<div class="empty-list">Your recorded scans will appear here.</div>';
    return;
  }
  scanList.innerHTML = scans
    .map(
      (scan) => `
        <button class="scan-card ${scan.id === selectedId ? "selected" : ""}" data-scan-id="${escapeHtml(scan.id)}" type="button" title="Press and hold to rename" aria-label="${escapeHtml(scan.name)}. Tap to open; press and hold to rename.">
          <span class="scan-card-top">
            <strong>${escapeHtml(scan.name)}</strong>
            <span class="status-dot ${escapeHtml(scan.status)}"></span>
          </span>
          <small>${escapeHtml(statusLabels[scan.status] || scan.status)} · ${escapeHtml(formatDate(scan.created_at))}</small>
        </button>`,
    )
    .join("");
  scanList.querySelectorAll("[data-scan-id]").forEach((button) => {
    wireScanCard(button);
  });
}

function statusPanel(scan) {
  const progress = Math.round(Math.max(0, Math.min(1, Number(scan.progress || 0))) * 100);
  const running = ["uploading", "processing_frames", "ma_queued", "ma_running", "da3_queued", "da3_running"].includes(scan.status);
  return `
    <div class="status-panel">
      <div class="status-line">
        <span>${escapeHtml(scan.message || statusLabels[scan.status] || scan.status)}</span>
        <b>${running ? `${progress}%` : escapeHtml(statusLabels[scan.status] || scan.status)}</b>
      </div>
      ${running ? `<div class="progress-track"><span style="width:${progress}%"></span></div>` : ""}
      ${scan.error ? `<div class="error-box">${escapeHtml(scan.error)}</div>` : ""}
    </div>`;
}

function preparedSection(scan) {
  const prepared = scan.prepared;
  if (!prepared) return "";
  const warnings = prepared.quality_warning_counts || {};
  const warningHtml = Object.entries(warnings)
    .filter(([, count]) => Number(count) > 0)
    .map(([name, count]) => `<span class="warning-chip">${escapeHtml(name.replaceAll("_", " "))}: ${Number(count)}</span>`)
    .join("");
  const thumbs = (prepared.frames || []).slice(0, 24).map((frame) => {
    const flagged = frame.quality?.warnings?.length ? "flagged" : "";
    return `<a class="prepared-thumb ${flagged}" href="${frame.frame_url}" target="_blank" rel="noopener">
      <img src="${frame.thumbnail_url}" alt="Prepared view ${Number(frame.index) + 1}" loading="lazy" />
      <span>#${Number(frame.index) + 1} · ${Number(frame.timestamp_s).toFixed(1)}s</span>
    </a>`;
  }).join("");
  return `
    <div class="stat-grid">
      <div class="stat"><b>${Number(prepared.frame_count)}</b><span>Prepared views</span></div>
      <div class="stat"><b>${Number(prepared.effective_fps).toFixed(2)}</b><span>Views / second</span></div>
      <div class="stat"><b>${formatDuration(prepared.probe?.duration_s)}</b><span>Video duration</span></div>
      <div class="stat"><b>${escapeHtml(prepared.probe?.codec || "—")}</b><span>Video codec</span></div>
    </div>
    ${warningHtml ? `<div class="warning-row">${warningHtml}</div>` : ""}
    <div class="media-panel">
      <img src="${prepared.contact_sheet_url}" alt="Prepared frame contact sheet" loading="lazy" />
      <div class="media-caption"><span>Prepared view coverage</span><a href="${prepared.manifest_url}" target="_blank" rel="noopener">Frame manifest</a></div>
    </div>
    <div class="subheading"><div><span class="eyebrow">Prepared input</span><h3>Sampled views</h3></div><span class="status-badge">First ${Math.min(24, prepared.frame_count)} shown</span></div>
    <div class="thumb-grid">${thumbs}</div>`;
}

function outputFrameCards(outputs) {
  const frames = outputs.frames || [];
  const pageSize = 8;
  const pageCount = Math.max(1, Math.ceil(frames.length / pageSize));
  outputPage = Math.min(outputPage, pageCount - 1);
  const first = outputPage * pageSize;
  const visible = frames.slice(first, first + pageSize);
  const cards = visible.map((frame) => `
    <article class="output-card">
      <div class="output-card-head">
        <b>View ${Number(frame.index) + 1}</b>
        <span>${Number(frame.timestamp_s || 0).toFixed(1)}s · ${Number(frame.depth?.p50_m || 0).toFixed(2)}m median</span>
      </div>
      <div class="output-images">
        <a class="output-image" href="${frame.model_rgb_url}" target="_blank" rel="noopener"><img src="${frame.model_rgb_url}" alt="Model RGB" loading="lazy" /><span>Model RGB</span></a>
        <a class="output-image" href="${frame.depth_preview_url}" target="_blank" rel="noopener"><img src="${frame.depth_preview_url}" alt="Depth" loading="lazy" /><span>Depth</span></a>
        <a class="output-image" href="${frame.confidence_preview_url}" target="_blank" rel="noopener"><img src="${frame.confidence_preview_url}" alt="Confidence" loading="lazy" /><span>Confidence</span></a>
        <a class="output-image" href="${frame.mask_preview_url}" target="_blank" rel="noopener"><img src="${frame.mask_preview_url}" alt="Validity mask" loading="lazy" /><span>Mask</span></a>
      </div>
      <div class="output-card-foot"><span>${(Number(frame.depth?.valid_fraction || 0) * 100).toFixed(1)}% valid depth</span><a href="${frame.raw_npz_url}" download>Raw NPZ</a></div>
    </article>`).join("");
  return `
    <div class="output-grid">${cards}</div>
    <div class="pager">
      <button id="output-prev" class="ghost-button" type="button" ${outputPage <= 0 ? "disabled" : ""}>Previous</button>
      <span>Page ${outputPage + 1} of ${pageCount}</span>
      <button id="output-next" class="ghost-button" type="button" ${outputPage >= pageCount - 1 ? "disabled" : ""}>Next</button>
    </div>`;
}

function alignmentSection(scan) {
  const alignment = scan.alignment;
  if (!alignment) {
    return `<div class="ready-callout alignment-callout"><div><h3>Register this walk to Noesis.</h3><p>The tool will preserve metric scale and gravity, fit the phone room structure to the fixed living-room camera reconstruction, and reject ambiguous fits.</p></div><button id="align-noesis" class="primary-button" type="button">Align to Noesis</button></div>`;
  }
  if (["queued", "running"].includes(alignment.status)) {
    const progress = Math.round(Math.max(0, Math.min(1, Number(alignment.progress || 0))) * 100);
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Aligning to Noesis")}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div></div>`;
  }
  if (alignment.status === "failed") {
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Noesis alignment failed")}</span><b>Not aligned</b></div><div class="error-box">${escapeHtml(alignment.error || "The automatic fit did not pass its quality gate.")}</div><div class="artifact-row"><button id="align-noesis" class="primary-button" type="button">Try alignment again</button></div></div>`;
  }
  const results = alignment.results;
  if (alignment.status !== "complete" || !results) return "";
  const urls = results.artifact_urls || {};
  const vertical = results.vertical_structure || {};
  const full = results.full_cloud || {};
  const reprojection = results.fixed_camera_reprojection || {};
  const links = [
    [urls.aligned_phone_glb, "Aligned RGB GLB"],
    [urls.comparison_glb, "Noesis comparison GLB"],
    [urls.transform, "MA → Noesis transform"],
    [urls.trajectory, "Aligned camera poses"],
    [urls.camera_solution_npz, "Aligned solution NPZ"],
    [urls.report, "Quality report"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  return `
    <div class="subheading"><div><span class="eyebrow">Noesis registration</span><h3>Living-room alignment passed</h3></div><span class="status-badge aligned-badge">Backend world · metric</span></div>
    <p class="alignment-note">This is a saved, quality-gated review candidate. It has not changed or promoted the live Noesis world.</p>
    <div class="stat-grid">
      <div class="stat"><b>${(Number(vertical.source_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Phone structure match</span></div>
      <div class="stat"><b>${(Number(vertical.target_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Fixed-view coverage</span></div>
      <div class="stat"><b>${Number(vertical.plane_residual_median_m || 0).toFixed(3)}m</b><span>Wall residual</span></div>
      <div class="stat"><b>${(Number(full.source_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Full cloud match</span></div>
    </div>
    <div class="artifact-row">${links}</div>
    <div class="alignment-evidence-grid">
      <div class="media-panel"><a href="${urls.topdown_preview}" target="_blank" rel="noopener"><img src="${urls.topdown_preview}" alt="Top-down Noesis and phone alignment" loading="lazy" /></a><div class="media-caption"><span>Blue: fixed-camera Noesis · orange: phone · green: walk</span><a href="${urls.topdown_preview}" target="_blank" rel="noopener">Full size</a></div></div>
      <div class="media-panel"><a href="${urls.fixed_camera_reprojection}" target="_blank" rel="noopener"><img src="${urls.fixed_camera_reprojection}" alt="Aligned phone reconstruction reprojected through fixed camera" loading="lazy" /></a><div class="media-caption"><span>Phone RGB projected through the calibrated security camera</span><span>${(Number(reprojection.target_cell_coverage_fraction || 0) * 100).toFixed(1)}% target-cell coverage</span></div></div>
    </div>`;
}

function outputsSection(scan) {
  const outputs = scan.outputs;
  if (!outputs) return "";
  const urls = outputs.artifact_urls || {};
  const aligned = scan.alignment?.status === "complete" ? scan.alignment.results : null;
  const alignedUrls = aligned?.artifact_urls || {};
  const providerLabel = outputs.provider === "da3" ? "DA3" : "MapAnything";
  const viewerTitle = aligned ? "Noesis-aligned comparison" : `${providerLabel} reconstruction`;
  const viewerBadge = aligned ? "Aligned backend world frame" : `Unaligned ${providerLabel} world frame`;
  const scale = outputs.scale || {};
  const files = outputs.files || [];
  const artifactLinks = [
    [urls.reconstruction_glb, "Reconstruction GLB"],
    [urls.trajectory_preview, "Camera path PNG"],
    [urls.trajectory_json, "Camera poses JSON"],
    [urls.camera_solution_npz, "Camera solution NPZ"],
    [urls.manifest, "Output manifest"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  const fileRows = files.map((file) => `<div class="file-row"><a href="${file.url}" target="_blank" rel="noopener">${escapeHtml(file.path)}</a><span>${formatBytes(file.size_bytes)}</span></div>`).join("");
  return `
    <div class="stat-grid">
      <div class="stat"><b>${Number(outputs.view_count)}</b><span>Joint views</span></div>
      <div class="stat"><b>${Number(outputs.review_point_count).toLocaleString()}</b><span>Review points</span></div>
      <div class="stat"><b>${Number(scale.median || 0).toFixed(3)}</b><span>Median metric scale</span></div>
      <div class="stat"><b>${files.length}</b><span>Saved output files</span></div>
    </div>
    ${alignmentSection(scan)}
    <div class="subheading"><div><span class="eyebrow">3D review</span><h3>${viewerTitle}</h3></div><span class="status-badge">${viewerBadge}</span></div>
    <div class="viewer-wrap">
      <div id="viewer-canvas" class="viewer-canvas"></div>
      <div id="viewer-overlay" class="viewer-overlay">Loading the saved GLB…</div>
      <div class="viewer-hint">One finger rotates · two fingers pan and zoom</div>
    </div>
    <div class="artifact-row">${alignedUrls.comparison_glb ? `<a class="artifact-link" href="${alignedUrls.comparison_glb}" target="_blank" rel="noopener">Current 3D comparison</a>` : ""}${artifactLinks}</div>
    <div class="media-panel">
      <img src="${urls.trajectory_preview}" alt="Top-down camera trajectory" loading="lazy" />
      <div class="media-caption"><span>Green is the first camera pose; orange is the walk path.</span><a href="${urls.trajectory_json}" target="_blank" rel="noopener">Pose data</a></div>
    </div>
    <div class="subheading"><div><span class="eyebrow">Per-view outputs</span><h3>RGB, depth, confidence, mask, and raw arrays</h3></div></div>
    <div id="output-pages">${outputFrameCards(outputs)}</div>
    <details class="files-panel"><summary>All ${files.length} saved output files</summary><div class="file-list">${fileRows}</div></details>`;
}

function disposeViewer() {
  if (!currentViewer) return;
  cancelAnimationFrame(currentViewer.animation);
  currentViewer.resizeObserver?.disconnect();
  currentViewer.controls?.dispose();
  currentViewer.scene?.traverse((object) => {
    object.geometry?.dispose?.();
    if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose?.());
    else object.material?.dispose?.();
  });
  currentViewer.renderer?.dispose();
  currentViewer = null;
}

function initializeViewer(glbUrl) {
  const host = document.querySelector("#viewer-canvas");
  const overlay = document.querySelector("#viewer-overlay");
  if (!host || !glbUrl) return;
  disposeViewer();
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0a);
  const camera = new THREE.PerspectiveCamera(52, 1, 0.001, 10000);
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.screenSpacePanning = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x223329, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(2, 4, 3);
  scene.add(sun);

  const resize = () => {
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  resize();

  let animation = 0;
  const tick = () => {
    controls.update();
    renderer.render(scene, camera);
    animation = requestAnimationFrame(tick);
    if (currentViewer) currentViewer.animation = animation;
  };
  currentViewer = { scene, camera, renderer, controls, resizeObserver, animation };
  tick();

  new GLTFLoader().load(
    glbUrl,
    (gltf) => {
      const root = gltf.scene;
      scene.add(root);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      const diagonal = Math.max(size.length(), 0.1);
      root.traverse((object) => {
        if (object.isPoints && object.material) {
          object.material.size = diagonal * 0.0017;
          object.material.sizeAttenuation = true;
          object.material.vertexColors = true;
          object.material.needsUpdate = true;
        }
      });
      const gridSize = Math.max(size.x, size.z, 1);
      const grid = new THREE.GridHelper(gridSize * 1.25, 20, 0x47675a, 0x25352f);
      grid.position.y = box.min.y;
      scene.add(grid);
      controls.target.copy(center);
      camera.near = Math.max(diagonal / 10000, 0.001);
      camera.far = diagonal * 100;
      camera.position.set(center.x + diagonal * 0.72, center.y + diagonal * 0.48, center.z + diagonal * 0.72);
      camera.updateProjectionMatrix();
      controls.update();
      overlay.textContent = `${diagonal.toFixed(2)}m scene diagonal · drag to inspect`;
    },
    undefined,
    (error) => {
      overlay.textContent = `3D viewer could not load: ${error?.message || "unknown error"}`;
    },
  );
}

function wireDetailActions(scan) {
  document.querySelector("#rename-scan")?.addEventListener("click", () => {
    void renameScan(scan.id);
  });
  document.querySelector("#delete-scan")?.addEventListener("click", async () => {
    const accepted = window.confirm(`Permanently delete “${scan.name}” and every saved video, frame, and reconstruction output?`);
    if (!accepted) return;
    try {
      await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}`, { method: "DELETE" });
      scans = scans.filter((item) => item.id !== scan.id);
      if (scans.length) rememberSelectedScan(scans[0].id);
      else {
        selectedId = null;
        newWalkMode = true;
        localStorage.removeItem("phoneScanSelected");
        localStorage.setItem(NEW_WALK_MODE_KEY, "1");
        syncCaptureModeUi();
      }
      lastDetailFingerprint = "";
      disposeViewer();
      renderScanList();
      renderSelected();
      toastMessage("Scan deleted from this machine");
    } catch (error) {
      toastMessage(`Delete failed: ${error.message}`);
    }
  });
  document.querySelector("#initiate-inference")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    const provider = document.querySelector("#inference-provider")?.value || "mapanything";
    const label = provider === "da3" ? "DA3" : "MapAnything";
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/initiate-inference?provider=${encodeURIComponent(provider)}`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage(`${label} started`);
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`${label} could not start: ${error.message}`);
    }
  });
  document.querySelector("#align-noesis")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/align-noesis`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage("Noesis alignment started");
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`Noesis alignment could not start: ${error.message}`);
    }
  });
  document.querySelector("#output-prev")?.addEventListener("click", () => {
    outputPage = Math.max(0, outputPage - 1);
    lastDetailFingerprint = "";
    renderSelected();
  });
  document.querySelector("#output-next")?.addEventListener("click", () => {
    outputPage += 1;
    lastDetailFingerprint = "";
    renderSelected();
  });
}

function renderSelected() {
  const scan = scans.find((item) => item.id === selectedId);
  if (!scan) {
    disposeViewer();
    if (newWalkMode) {
      scanDetail.replaceChildren();
      setScanDetailVisible(false);
      return;
    }
    setScanDetailVisible(true);
    scanDetail.innerHTML = `<div class="empty-state"><div class="empty-orbit"><span></span></div><h2>No walk selected</h2><p>Record a room walk above or select a saved walk.</p></div>`;
    return;
  }
  setScanDetailVisible(true);
  const fingerprint = JSON.stringify(scan);
  if (fingerprint === lastDetailFingerprint) return;
  lastDetailFingerprint = fingerprint;
  disposeViewer();
  const alignmentRunning = ["queued", "running"].includes(scan.alignment?.status);
  const canDelete = !["uploading", "processing_frames", "ma_queued", "ma_running", "da3_queued", "da3_running"].includes(scan.status) && !alignmentRunning;
  const canInitiate = ["ready", "ma_failed", "da3_failed"].includes(scan.status);
  const videoSize = formatBytes(scan.video?.size_bytes);
  scanDetail.innerHTML = `
    <div class="detail-header">
      <div class="detail-title">
        <span class="eyebrow">${escapeHtml(statusLabels[scan.status] || scan.status)}</span>
        <h2>${escapeHtml(scan.name)}</h2>
        <div class="detail-meta"><span>${escapeHtml(formatDate(scan.created_at))}</span><span>${videoSize}</span><span>${escapeHtml(scan.id)}</span></div>
      </div>
      <div class="detail-actions">
        <a class="ghost-button" href="${scan.video?.url}" target="_blank" rel="noopener">Original video</a>
        <button id="rename-scan" class="ghost-button" type="button">Rename</button>
        ${canDelete ? '<button id="delete-scan" class="danger-button" type="button">Delete scan</button>' : ""}
      </div>
    </div>
    ${statusPanel(scan)}
    ${canInitiate ? `<div class="ready-callout inference-callout"><div><h3>${scan.status.endsWith("_failed") ? "The prepared views are still safe." : "The room walk is ready."}</h3><p>Both options jointly reconstruct all ${Number(scan.prepared?.frame_count || 0)} phone views. DA3 uses DA3-BASE any-view geometry plus the validated DA3Metric-Large FP16 TensorRT engine for metric scale.</p></div><label class="provider-picker"><span>Provider</span><select id="inference-provider"><option value="mapanything" ${scan.provider !== "da3" ? "selected" : ""}>MapAnything</option><option value="da3" ${scan.provider === "da3" ? "selected" : ""}>DA3</option></select></label><button id="initiate-inference" class="primary-button" type="button">Run reconstruction</button></div>` : ""}
    ${scan.status === "complete" ? outputsSection(scan) : preparedSection(scan)}
    ${scan.status !== "complete" && scan.video?.url ? `<div class="subheading"><div><span class="eyebrow">Source</span><h3>Original phone video</h3></div></div><div class="media-panel"><video src="${scan.video.url}" controls preload="metadata" playsinline></video><div class="media-caption"><span>${escapeHtml(scan.video.original_name || "phone video")}</span><span>${videoSize}</span></div></div>` : ""}`;
  wireDetailActions(scan);
  if (scan.status === "complete") {
    const viewerUrl = scan.alignment?.status === "complete"
      ? scan.alignment?.results?.artifact_urls?.comparison_glb
      : scan.outputs?.artifact_urls?.reconstruction_glb;
    requestAnimationFrame(() => initializeViewer(viewerUrl));
  }
}

async function refreshScans({ force = false } = {}) {
  if (refreshing) return;
  refreshing = true;
  try {
    scans = await jsonFetch("/api/scans");
    if (selectedId && !scans.some((scan) => scan.id === selectedId)) {
      selectedId = null;
      localStorage.removeItem("phoneScanSelected");
    }
    if (!selectedId && scans.length && !newWalkMode) {
      rememberSelectedScan(scans[0].id);
    }
    if (force) lastDetailFingerprint = "";
    renderScanList();
    renderSelected();
  } catch (error) {
    toastMessage(`Could not refresh scans: ${error.message}`);
  } finally {
    refreshing = false;
  }
}

function uploadVideo(file) {
  if (!file) return;
  if (uploadInProgress) {
    toastMessage("A walk is already uploading");
    return;
  }
  uploadInProgress = true;
  uploadPanel.classList.remove("hidden");
  uploadLabel.textContent = `Uploading ${file.name || "phone video"}`;
  uploadPercent.textContent = "0%";
  uploadBar.style.width = "0%";
  const xhr = new XMLHttpRequest();
  const name = scanName.value.trim() || "Phone room walk";
  xhr.open("POST", `/api/scans?name=${encodeURIComponent(name)}`);
  xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
  xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name || "phone_walk.mp4"));
  xhr.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    uploadPercent.textContent = `${percent}%`;
    uploadBar.style.width = `${percent}%`;
  });
  xhr.addEventListener("load", async () => {
    uploadInProgress = false;
    cameraInput.value = "";
    existingInput.value = "";
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = `${xhr.status} upload failed`;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) { /* keep HTTP detail */ }
      uploadLabel.textContent = detail;
      toastMessage(detail);
      return;
    }
    const scan = JSON.parse(xhr.responseText);
    rememberSelectedScan(scan.id);
    uploadPercent.textContent = "100%";
    uploadBar.style.width = "100%";
    uploadLabel.textContent = "Upload saved · preparing multi-view frames";
    toastMessage("Walk auto-saved; frame preparation started");
    await refreshScans({ force: true });
    setTimeout(() => uploadPanel.classList.add("hidden"), 1800);
  });
  xhr.addEventListener("error", () => {
    uploadInProgress = false;
    uploadLabel.textContent = "Upload connection failed";
    toastMessage("Upload failed. Keep this page open and confirm the phone is still on the home Wi-Fi.");
  });
  xhr.addEventListener("abort", () => {
    uploadInProgress = false;
  });
  xhr.send(file);
}

cameraInput.addEventListener("change", () => uploadVideo(cameraInput.files?.[0]));
existingInput.addEventListener("change", () => uploadVideo(existingInput.files?.[0]));
refreshButton.addEventListener("click", () => refreshScans({ force: true }));
newWalkButton.addEventListener("click", startNewWalk);

if (newWalkMode) localStorage.removeItem("phoneScanSelected");
syncCaptureModeUi();
refreshHealth();
refreshScans({ force: true });
setInterval(refreshHealth, 30_000);
setInterval(refreshScans, 2_000);
