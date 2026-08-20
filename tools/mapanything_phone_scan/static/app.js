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
let alignmentTargets = [];
let alignmentReleaseId = null;
let pcfPausesAppliance = false;

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

const supplementStatusLabels = {
  uploading: "Uploading video",
  processing_frames: "Preparing new views",
  ready: "Ready to integrate",
  frame_failed: "Frame preparation failed",
  queued: "Integration queued",
  running: "Registering and merging",
  integration_failed: "Not merged",
  complete: "Merged revision saved",
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

function alignmentTargetFor(cameraId) {
  return alignmentTargets.find((target) => target.camera_id === cameraId) || null;
}

function cameraLabel(cameraId) {
  const configured = alignmentTargetFor(cameraId)?.label;
  if (configured) return configured;
  return String(cameraId || "Unknown camera")
    .replaceAll("_", "-")
    .split("-")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function alignmentTargetControls(selectedCameraId, buttonLabel) {
  const selectedIsAvailable = Boolean(alignmentTargetFor(selectedCameraId));
  const options = alignmentTargets.map((target) => `
    <option value="${escapeHtml(target.camera_id)}" ${target.camera_id === selectedCameraId ? "selected" : ""}>
      ${escapeHtml(target.label)} camera
    </option>`).join("");
  const releaseNote = alignmentReleaseId
    ? `Validated scene release: ${escapeHtml(alignmentReleaseId)}`
    : "No validated scene release is available";
  return `
    <div class="alignment-actions">
      <label class="provider-picker alignment-picker">
        <span>Static camera for this room</span>
        <select id="alignment-target" ${alignmentTargets.length ? "" : "disabled"}>
          <option value="" ${selectedIsAvailable ? "" : "selected"}>Choose camera…</option>
          ${options}
        </select>
        <small>${releaseNote}</small>
      </label>
      <button id="align-noesis" class="primary-button" type="button" ${selectedIsAvailable ? "" : "disabled"}>${escapeHtml(buttonLabel)}</button>
    </div>`;
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
    healthPill.textContent = `${health.device} · adaptive views · ${health.max_selected_frames} emergency max`;
    healthPill.className = "health-pill online";
    const nextTargets = Array.isArray(health.alignment_targets)
      ? health.alignment_targets.filter((target) => target?.camera_id && target?.revision_id)
      : [];
    const nextReleaseId = health.alignment_release_id || null;
    const nextPcfPausesAppliance = health.pcf_pauses_appliance === true;
    const alignmentConfigChanged = JSON.stringify([
      alignmentTargets,
      alignmentReleaseId,
      pcfPausesAppliance,
    ]) !== JSON.stringify([
      nextTargets,
      nextReleaseId,
      nextPcfPausesAppliance,
    ]);
    alignmentTargets = nextTargets;
    alignmentReleaseId = nextReleaseId;
    pcfPausesAppliance = nextPcfPausesAppliance;
    if (alignmentConfigChanged) {
      lastDetailFingerprint = "";
      renderSelected();
    }
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
  const selection = prepared.selection || {};
  const warnings = prepared.quality_warning_counts || {};
  const warningHtml = Object.entries(warnings)
    .filter(([, count]) => Number(count) > 0)
    .map(([name, count]) => `<span class="warning-chip">${escapeHtml(name.replaceAll("_", " "))}: ${Number(count)}</span>`)
    .join("");
  const thumbs = (prepared.frames || []).slice(0, 24).map((frame) => {
    const flagged = frame.quality?.warnings?.length ? "flagged" : "";
    return `<a class="prepared-thumb ${flagged}" href="${frame.frame_url}" target="_blank" rel="noopener">
      <img src="${frame.thumbnail_url}" alt="Prepared view ${Number(frame.index) + 1}" loading="lazy" />
      <span>#${Number(frame.index) + 1} · ${Number(frame.timestamp_s).toFixed(1)}s${frame.selection_reason ? ` · ${escapeHtml(frame.selection_reason.replaceAll("_", " "))}` : ""}</span>
    </a>`;
  }).join("");
  return `
    <div class="stat-grid">
      <div class="stat"><b>${Number(prepared.frame_count)}</b><span>Prepared views</span></div>
      <div class="stat"><b>${Number(prepared.candidate_count || prepared.frame_count)}</b><span>Analyzed candidates</span></div>
      <div class="stat"><b>${formatDuration(prepared.probe?.duration_s)}</b><span>Video duration</span></div>
      <div class="stat"><b>${Number(selection.adjacent_connectivity_pass_fraction ?? 0).toLocaleString(undefined, { style: "percent", maximumFractionDigits: 0 })}</b><span>Connected view pairs</span></div>
    </div>
    ${selection.selection_limited ? '<div class="warning-row"><span class="warning-chip">Emergency view ceiling reached</span></div>' : ""}
    ${warningHtml ? `<div class="warning-row">${warningHtml}</div>` : ""}
    <div class="media-panel">
      <img src="${prepared.contact_sheet_url}" alt="Prepared frame contact sheet" loading="lazy" />
      <div class="media-caption"><span>Prepared view coverage</span><a href="${prepared.manifest_url}" target="_blank" rel="noopener">Frame manifest</a></div>
    </div>
    <div class="subheading"><div><span class="eyebrow">Prepared input</span><h3>Adaptive reconstruction views</h3></div><span class="status-badge">First ${Math.min(24, prepared.frame_count)} shown</span></div>
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
    return `<div class="ready-callout alignment-callout"><div><h3>Register this walk to the correct room camera.</h3><p>Choose the static camera physically installed in this room. The tool will preserve metric scale and gravity, fit against that camera's validated reconstruction, and reject ambiguous fits.</p></div>${alignmentTargetControls("", "Align to selected camera")}</div>`;
  }
  if (["queued", "running"].includes(alignment.status)) {
    const progress = Math.round(Math.max(0, Math.min(1, Number(alignment.progress || 0))) * 100);
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Aligning to Noesis")}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div><p class="alignment-target-note">Target: ${escapeHtml(cameraLabel(alignment.target_camera_id))} · ${escapeHtml(alignment.target_revision_id || "validated revision")}</p></div>`;
  }
  if (alignment.status === "failed") {
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Noesis alignment failed")}</span><b>Not aligned</b></div><div class="error-box">${escapeHtml(alignment.error || "The automatic fit did not pass its quality gate.")}</div><p class="alignment-target-note">Previous target: ${escapeHtml(cameraLabel(alignment.target_camera_id))}</p>${alignmentTargetControls(alignment.target_camera_id || "", "Try selected camera")}</div>`;
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
    [urls.transform, "Phone → Noesis transform"],
    [urls.trajectory, "Aligned camera poses"],
    [urls.camera_solution_npz, "Aligned solution NPZ"],
    [urls.report, "Quality report"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  return `
    <div class="subheading"><div><span class="eyebrow">Noesis registration</span><h3>${escapeHtml(cameraLabel(results.target_camera_id || alignment.target_camera_id))} alignment passed</h3></div><span class="status-badge aligned-badge">Backend world · metric</span></div>
    <p class="alignment-note">This is a saved, quality-gated review candidate aligned against ${escapeHtml(results.target_revision_id || alignment.target_revision_id || "the selected revision")}. It has not changed or promoted the live Noesis world.</p>
    <div class="stat-grid">
      <div class="stat"><b>${(Number(vertical.source_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Camera-visible structure match</span></div>
      <div class="stat"><b>${(Number(vertical.target_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Fixed-view coverage</span></div>
      <div class="stat"><b>${Number(vertical.plane_residual_median_m || 0).toFixed(3)}m</b><span>Wall residual</span></div>
      <div class="stat"><b>${(Number(full.source_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Camera-visible cloud match</span></div>
    </div>
    <div class="artifact-row">${links}</div>
    <div class="alignment-evidence-grid">
      <div class="media-panel"><a href="${urls.topdown_preview}" target="_blank" rel="noopener"><img src="${urls.topdown_preview}" alt="Top-down Noesis and phone alignment" loading="lazy" /></a><div class="media-caption"><span>Blue: fixed-camera Noesis · orange: phone · green: walk</span><a href="${urls.topdown_preview}" target="_blank" rel="noopener">Full size</a></div></div>
      <div class="media-panel"><a href="${urls.fixed_camera_reprojection}" target="_blank" rel="noopener"><img src="${urls.fixed_camera_reprojection}" alt="Aligned phone reconstruction reprojected through fixed camera" loading="lazy" /></a><div class="media-caption"><span>Phone RGB projected through the calibrated security camera</span><span>${(Number(reprojection.target_cell_coverage_fraction || 0) * 100).toFixed(1)}% target-cell coverage</span></div></div>
    </div>`;
}

function pcfSection(scan) {
  if (scan.alignment?.status !== "complete") return "";
  const provider = String(scan.provider || scan.outputs?.provider || "").toLowerCase();
  const pcf = scan.pcf;
  const hasActiveSupplement = Boolean(scan.active_revision?.supplement_id);
  if (!pcf && provider !== "da3") {
    return `<div class="ready-callout pcf-callout pcf-unavailable"><div><span class="eyebrow">Optional final stage</span><h3>Prior-Conditioned Fusion requires a DA3 base walk.</h3><p>This walk was reconstructed with MapAnything. Its existing outputs remain valid, but PCF will not silently substitute or overwrite the provider.</p></div><span class="status-badge">Unavailable for this walk</span></div>`;
  }
  if (!pcf && hasActiveSupplement) {
    return `<div class="ready-callout pcf-callout pcf-unavailable"><div><span class="eyebrow">Optional final stage</span><h3>PCF is not available for this merged revision yet.</h3><p>The current PCF contract requires one exact prepared-view set through DA3 and conditioned MapAnything. The added-video revision will not be silently omitted.</p></div><span class="status-badge">Base-walk PCF only</span></div>`;
  }
  if (!pcf) {
    const runtimeNote = pcfPausesAppliance
      ? " The native Noesis/Menon appliance will be paused for GPU headroom and restored automatically afterward."
      : "";
    return `<div class="ready-callout pcf-callout"><div><span class="eyebrow">Optional final stage</span><h3>Build a Prior-Conditioned Fusion review.</h3><p>Run conditioned MapAnything over these same adaptive views, consistency-gate it against DA3, and generate static-world diagnostics. This saves a review candidate only; it does not publish a Scene Prior.${runtimeNote}</p></div><button id="initiate-pcf" class="primary-button" type="button">Generate PCF review</button></div>`;
  }
  if (["queued", "running"].includes(pcf.status)) {
    const progress = Math.round(Math.max(0, Math.min(1, Number(pcf.progress || 0))) * 100);
    return `<div class="status-panel pcf-status"><div class="status-line"><span>${escapeHtml(pcf.message || "Building PCF review")}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div><p class="alignment-target-note">${escapeHtml(cameraLabel(pcf.target_camera_id))} · review-only candidate · original prepared walk</p></div>`;
  }
  if (pcf.status === "failed") {
    return `<div class="status-panel pcf-status"><div class="status-line"><span>${escapeHtml(pcf.message || "PCF failed")}</span><b>Not completed</b></div><div class="error-box">${escapeHtml(pcf.error || "The PCF run did not complete.")}</div><div class="artifact-row">${pcf.log_url ? `<a class="artifact-link" href="${pcf.log_url}" target="_blank" rel="noopener">Run log</a>` : ""}<button id="initiate-pcf" class="primary-button compact-button" type="button">Retry PCF</button></div></div>`;
  }
  const results = pcf.results;
  if (pcf.status !== "complete" || !results) return "";
  const urls = results.artifact_urls || {};
  const fusion = results.fusion || {};
  const surfels = results.surfel_fusion || {};
  const multiview = results.multiview_consistency?.consensus || {};
  const heldout = results.heldout_even_to_odd_reprojection?.consensus || {};
  const staticVisible = results.evaluation?.fixed_camera_visible_cloud_metrics || {};
  const links = [
    [urls.pcf_glb, "PCF surfel GLB"],
    [urls.conditioned_mapanything_glb, "Conditioned MapAnything GLB"],
    [urls.collaboration_diagnostics, "Provider collaboration"],
    [urls.consensus_manifest, "Consensus manifest"],
    [urls.conditioned_mapanything_manifest, "Conditioned MA manifest"],
    [urls.evaluation_metrics, "Evaluation metrics"],
    [urls.run_log, "Run log"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  const evidence = [
    [urls.diagnostic_layers, "Static-world diagnostic layers"],
    [urls.point_preserving_layers, "2.5 cm point-preserving layers"],
    [urls.static_alignment_topdown, "PCF and static-camera alignment"],
    [urls.fixed_camera_reprojection, "Fixed-camera reprojection"],
    [urls.collaboration_diagnostics, "MapAnything and DA3 collaboration"],
    [urls.evaluation_overview, "DA3, conditioned MA, and PCF overview"],
  ].filter(([url]) => url).map(([url, label]) => `<div class="media-panel"><a href="${url}" target="_blank" rel="noopener"><img src="${url}" alt="${escapeHtml(label)}" loading="lazy" /></a><div class="media-caption"><span>${escapeHtml(label)}</span><a href="${url}" target="_blank" rel="noopener">Full size</a></div></div>`).join("");
  return `
    <section class="pcf-results">
      <div class="subheading"><div><span class="eyebrow">Optional final stage</span><h3>Prior-Conditioned Fusion complete</h3></div><span class="status-badge aligned-badge">Review candidate · not published</span></div>
      <p class="alignment-note">PCF used the original ${Number(results.view_count || scan.outputs?.view_count || 0)}-view DA3 walk, its passed ${escapeHtml(cameraLabel(results.target_camera_id || pcf.target_camera_id))} alignment, conditioned MapAnything, and DA3-carried consistency fusion. Raw evidence and diagnostics are auto-saved separately from the walk.</p>
      ${results.runtime_restore_error ? `<div class="error-box">The PCF files are complete, but automatic appliance restoration needs attention: ${escapeHtml(results.runtime_restore_error)}</div>` : ""}
      <div class="stat-grid">
        <div class="stat"><b>${Number(surfels.surfel_count || 0).toLocaleString()}</b><span>PCF surfels</span></div>
        <div class="stat"><b>${(Number(fusion.agreement_fraction_of_both || 0) * 100).toFixed(1)}%</b><span>Provider agreement</span></div>
        <div class="stat"><b>${Number(multiview.p80_error_m || 0).toFixed(3)}m</b><span>Multiview p80</span></div>
        <div class="stat"><b>${(Number(heldout.odd_frame_valid_pixel_coverage_fraction || 0) * 100).toFixed(1)}%</b><span>Held-out coverage</span></div>
        <div class="stat"><b>${Number(heldout.even_frame_map_to_odd_frame_depth_median_m || 0).toFixed(3)}m</b><span>Held-out median</span></div>
        <div class="stat"><b>${(Number(staticVisible.source_overlap_0_30m || 0) * 100).toFixed(1)}%</b><span>Static-visible match</span></div>
      </div>
      <div class="artifact-row">${links}</div>
      <div class="pcf-evidence-grid">${evidence}</div>
    </section>`;
}

function supplementsSection(scan) {
  const supplements = Array.isArray(scan.supplements) ? scan.supplements : [];
  const providerLabel = scan.provider === "da3" || scan.outputs?.provider === "da3" ? "DA3" : "MapAnything";
  const cards = supplements.map((addition, index) => {
    const running = ["uploading", "processing_frames", "queued", "running"].includes(addition.status);
    const progress = Math.round(Math.max(0, Math.min(1, Number(addition.progress || 0))) * 100);
    const prepared = addition.prepared || {};
    const results = addition.results || {};
    const urls = results.artifact_urls || {};
    const fusion = results.fusion || {};
    const active = scan.active_revision?.supplement_id === addition.id;
    const links = [
      [urls.merged_reconstruction_glb, "Merged RGB GLB"],
      [urls.added_evidence_glb, "Added evidence GLB"],
      [urls.source_comparison_glb, "Blue/orange source comparison"],
      [urls.trajectory_preview, "Added camera path"],
      [urls.append_to_base_transform, "Registration transform"],
      [urls.report, "Integration report"],
      [urls.manifest, "Revision manifest"],
    ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
    const canDelete = !running;
    const canIntegrate = ["ready", "integration_failed"].includes(addition.status);
    return `
      <article class="supplement-card ${active ? "active" : ""}">
        <div class="supplement-head">
          <div>
            <span class="eyebrow">Additional pass ${index + 1}</span>
            <h4>${escapeHtml(supplementStatusLabels[addition.status] || addition.status)}</h4>
            <small>${escapeHtml(formatDate(addition.created_at))} · ${formatBytes(addition.video?.size_bytes)}</small>
          </div>
          <div class="supplement-actions">
            ${active ? '<span class="status-badge aligned-badge">Current revision</span>' : ""}
            <a class="ghost-button compact-button" href="${addition.video?.url}" target="_blank" rel="noopener">Video</a>
            ${canDelete ? `<button class="danger-button compact-button" type="button" data-delete-supplement="${escapeHtml(addition.id)}">Delete</button>` : ""}
          </div>
        </div>
        ${running ? `<div class="status-panel supplement-progress"><div class="status-line"><span>${escapeHtml(addition.message || supplementStatusLabels[addition.status])}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div></div>` : ""}
        ${addition.error ? `<div class="error-box">${escapeHtml(addition.error)}</div>` : ""}
        ${active && scan.active_revision?.noesis_derivative_error ? `<div class="error-box">Merged revision saved, but its Noesis-world derivative failed: ${escapeHtml(scan.active_revision.noesis_derivative_error)}</div>` : ""}
        ${prepared.contact_sheet_url && addition.status !== "complete" ? `<div class="supplement-ready-grid"><a class="supplement-sheet" href="${prepared.contact_sheet_url}" target="_blank" rel="noopener"><img src="${prepared.contact_sheet_url}" alt="Prepared views from additional video" loading="lazy" /></a><div><b>${Number(prepared.frame_count || 0)} new views prepared</b><p>The tool will choose exact old bridge views, jointly infer them with these frames, and publish only if duplicate-pose and independent RGB/3D checks agree.</p>${canIntegrate ? `<button class="primary-button" type="button" data-integrate-supplement="${escapeHtml(addition.id)}">${addition.status === "integration_failed" ? "Retry integration" : `Integrate with ${providerLabel}`}</button>` : ""}</div></div>` : ""}
        ${addition.status === "complete" ? `<div class="supplement-result-grid"><div class="stat"><b>${Number(fusion.point_count || 0).toLocaleString()}</b><span>Merged surfels</span></div><div class="stat"><b>${Number(fusion.new_only_voxel_count || 0).toLocaleString()}</b><span>New-only voxels</span></div><div class="stat"><b>${Number(results.bridge?.selected_count || 0)}</b><span>Bridge views</span></div><div class="stat"><b>${Number(results.independent_pnp_validation?.accepted_anchor_count || 0)}</b><span>Independent anchors</span></div></div><div class="artifact-row">${links}</div>` : ""}
      </article>`;
  }).join("");
  return `
    <section class="supplements-section">
      <div class="subheading supplement-title"><div><span class="eyebrow">Additive reconstruction</span><h3>Additional room videos</h3></div><label class="ghost-button compact-button" for="additional-existing-video">Use saved video</label></div>
      <p class="alignment-note">Each pass is auto-saved. The current reconstruction stays immutable; accepted additions create a new versioned revision in its coordinate frame.</p>
      <input id="additional-existing-video" class="visually-hidden" type="file" accept="video/*" />
      ${cards || '<div class="supplement-empty"><b>No added passes yet.</b><span>Use Add Video above and begin where the earlier walk has recognizable overlap.</span></div>'}
    </section>`;
}

function outputsSection(scan) {
  const outputs = scan.outputs;
  if (!outputs) return "";
  const urls = outputs.artifact_urls || {};
  const aligned = scan.alignment?.status === "complete" ? scan.alignment.results : null;
  const alignedUrls = aligned?.artifact_urls || {};
  const activeRevision = scan.active_revision || null;
  const activeUrls = activeRevision?.artifact_urls || {};
  const pcfResult = scan.pcf?.status === "complete" ? scan.pcf.results : null;
  const pcfUrls = pcfResult?.artifact_urls || {};
  const providerLabel = outputs.provider === "da3" ? "DA3" : "MapAnything";
  const viewerTitle = activeRevision ? "Current merged reconstruction" : (pcfResult ? "Prior-Conditioned Fusion reconstruction" : (aligned ? "Noesis-aligned comparison" : `${providerLabel} reconstruction`));
  const viewerBadge = activeRevision
    ? (activeUrls.noesis_aligned_glb ? "Merged · aligned backend world" : "Merged · original phone world")
    : (pcfResult ? "PCF surfels · DA3 carrier frame" : (aligned ? "Aligned backend world frame" : `Unaligned ${providerLabel} world frame`));
  const scale = outputs.scale || {};
  const files = outputs.files || [];
  const artifactLinks = [
    [urls.reconstruction_glb, "Reconstruction GLB"],
    [urls.trajectory_preview, "Camera path PNG"],
    [urls.trajectory_json, "Camera poses JSON"],
    [urls.camera_solution_npz, "Camera solution NPZ"],
    [urls.window_registration_report, "Window registration report"],
    [urls.manifest, "Output manifest"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  const fileRows = files.map((file) => `<div class="file-row"><a href="${file.url}" target="_blank" rel="noopener">${escapeHtml(file.path)}</a><span>${formatBytes(file.size_bytes)}</span></div>`).join("");
  return `
    <div class="stat-grid">
      <div class="stat"><b>${Number(outputs.view_count)}</b><span>Selected views</span></div>
      <div class="stat"><b>${Number(outputs.review_point_count).toLocaleString()}</b><span>Review points</span></div>
      <div class="stat"><b>${Number(outputs.window_count || 1)}</b><span>Joint inference windows</span></div>
      <div class="stat"><b>${Number(scale.median || 0).toFixed(3)}</b><span>Median metric scale</span></div>
    </div>
    ${supplementsSection(scan)}
    ${alignmentSection(scan)}
    ${pcfSection(scan)}
    <div class="subheading"><div><span class="eyebrow">3D review</span><h3>${viewerTitle}</h3></div><span class="status-badge">${viewerBadge}</span></div>
    <div class="viewer-wrap">
      <div id="viewer-canvas" class="viewer-canvas"></div>
      <div id="viewer-overlay" class="viewer-overlay">Loading the saved GLB…</div>
      <div class="viewer-hint">One finger rotates · two fingers pan and zoom</div>
    </div>
    <div class="artifact-row">${pcfUrls.pcf_glb ? `<a class="artifact-link" href="${pcfUrls.pcf_glb}" target="_blank" rel="noopener">PCF surfel reconstruction</a>` : ""}${activeUrls.source_comparison_glb ? `<a class="artifact-link" href="${activeUrls.source_comparison_glb}" target="_blank" rel="noopener">Current source comparison</a>` : ""}${alignedUrls.comparison_glb ? `<a class="artifact-link" href="${alignedUrls.comparison_glb}" target="_blank" rel="noopener">Noesis alignment comparison</a>` : ""}${artifactLinks}</div>
    <div class="media-panel">
      <img src="${activeUrls.trajectory_preview || urls.trajectory_preview}" alt="Top-down camera trajectory" loading="lazy" />
      <div class="media-caption"><span>Green is the first camera pose; orange is the walk path.</span><a href="${activeUrls.trajectory_json || urls.trajectory_json}" target="_blank" rel="noopener">Pose data</a></div>
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
  document.querySelector("#additional-camera-video")?.addEventListener("change", (event) => {
    void uploadSupplementVideo(scan, event.currentTarget.files?.[0]);
  });
  document.querySelector("#additional-existing-video")?.addEventListener("change", (event) => {
    void uploadSupplementVideo(scan, event.currentTarget.files?.[0]);
  });
  document.querySelectorAll("[data-integrate-supplement]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      const supplementId = event.currentTarget.dataset.integrateSupplement;
      if (!supplementId) return;
      const providerLabel = scan.provider === "da3" || scan.outputs?.provider === "da3" ? "DA3" : "MapAnything";
      const accepted = window.confirm(`Run a joint ${providerLabel} bridge reconstruction and merge this added pass if its registration checks pass?`);
      if (!accepted) return;
      event.currentTarget.disabled = true;
      try {
        const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/supplements/${encodeURIComponent(supplementId)}/integrate`, { method: "POST" });
        const index = scans.findIndex((item) => item.id === scan.id);
        if (index >= 0) scans[index] = updated;
        lastDetailFingerprint = "";
        renderSelected();
        toastMessage(`${providerLabel} bridge integration started`);
      } catch (error) {
        event.currentTarget.disabled = false;
        toastMessage(`Integration could not start: ${error.message}`);
      }
    });
  });
  document.querySelectorAll("[data-delete-supplement]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      const supplementId = event.currentTarget.dataset.deleteSupplement;
      if (!supplementId) return;
      const accepted = window.confirm("Delete this added video, its prepared frames, and its revision outputs? The original room walk will remain untouched.");
      if (!accepted) return;
      event.currentTarget.disabled = true;
      try {
        await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/supplements/${encodeURIComponent(supplementId)}`, { method: "DELETE" });
        toastMessage("Added video and its revision were deleted");
        await refreshScans({ force: true });
      } catch (error) {
        event.currentTarget.disabled = false;
        toastMessage(`Added video could not be deleted: ${error.message}`);
      }
    });
  });
  const alignmentTarget = document.querySelector("#alignment-target");
  const alignmentButton = document.querySelector("#align-noesis");
  alignmentTarget?.addEventListener("change", () => {
    if (alignmentButton) alignmentButton.disabled = !alignmentTarget.value;
  });
  alignmentButton?.addEventListener("click", async (event) => {
    const cameraId = alignmentTarget?.value || "";
    const target = alignmentTargetFor(cameraId);
    if (!target) {
      toastMessage("Choose the static camera for this room before aligning");
      event.currentTarget.disabled = true;
      return;
    }
    const accepted = window.confirm(
      `Align “${scan.name}” to the ${target.label} camera reconstruction?`,
    );
    if (!accepted) return;
    event.currentTarget.disabled = true;
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/align-noesis?camera_id=${encodeURIComponent(cameraId)}`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage(`Alignment to ${target.label} started`);
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`Noesis alignment could not start: ${error.message}`);
    }
  });
  document.querySelector("#initiate-pcf")?.addEventListener("click", async (event) => {
    const runtimeSentence = pcfPausesAppliance
      ? " The native Noesis/Menon appliance will be temporarily paused for GPU headroom and restored automatically when the job exits."
      : "";
    const accepted = window.confirm(
      `Generate the review-only PCF candidate now? This runs conditioned MapAnything, DA3 consistency fusion, and static-world evaluation. It can take a while and needs substantial GPU headroom and storage.${runtimeSentence}`,
    );
    if (!accepted) return;
    event.currentTarget.disabled = true;
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/initiate-pcf`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage("PCF review started; the page can remain open while it runs");
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`PCF could not start: ${error.message}`);
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
  const supplementRunning = (scan.supplements || []).some((addition) => ["uploading", "processing_frames", "queued", "running"].includes(addition.status));
  const pcfRunning = ["queued", "running"].includes(scan.pcf?.status);
  const canDelete = !["uploading", "processing_frames", "ma_queued", "ma_running", "da3_queued", "da3_running"].includes(scan.status) && !alignmentRunning && !supplementRunning && !pcfRunning;
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
        ${scan.status === "complete" && !supplementRunning && !pcfRunning ? '<label class="secondary-button" for="additional-camera-video">＋ Add Video</label><input id="additional-camera-video" class="visually-hidden" type="file" accept="video/*" capture="environment" />' : ""}
        <a class="ghost-button" href="${scan.video?.url}" target="_blank" rel="noopener">Original video</a>
        <button id="rename-scan" class="ghost-button" type="button">Rename</button>
        ${canDelete ? '<button id="delete-scan" class="danger-button" type="button">Delete scan</button>' : ""}
      </div>
    </div>
    ${statusPanel(scan)}
    ${canInitiate ? `<div class="ready-callout inference-callout"><div><h3>${scan.status.endsWith("_failed") ? "The prepared views are still safe." : "The room walk is ready."}</h3><p>Reconstruction uses all ${Number(scan.prepared?.frame_count || 0)} adaptive phone views. MapAnything automatically uses overlapping registered windows above its measured joint-view capacity. DA3 uses DA3-BASE any-view geometry plus the validated DA3Metric-Large FP16 TensorRT engine for metric scale.</p></div><label class="provider-picker"><span>Provider</span><select id="inference-provider"><option value="mapanything" ${scan.provider !== "da3" ? "selected" : ""}>MapAnything</option><option value="da3" ${scan.provider === "da3" ? "selected" : ""}>DA3</option></select></label><button id="initiate-inference" class="primary-button" type="button">Run reconstruction</button></div>` : ""}
    ${scan.status === "complete" ? outputsSection(scan) : preparedSection(scan)}
    ${scan.status !== "complete" && scan.video?.url ? `<div class="subheading"><div><span class="eyebrow">Source</span><h3>Original phone video</h3></div></div><div class="media-panel"><video src="${scan.video.url}" controls preload="metadata" playsinline></video><div class="media-caption"><span>${escapeHtml(scan.video.original_name || "phone video")}</span><span>${videoSize}</span></div></div>` : ""}`;
  wireDetailActions(scan);
  if (scan.status === "complete") {
    const revisionUrls = scan.active_revision?.artifact_urls || {};
    const pcfUrls = scan.pcf?.status === "complete" ? scan.pcf.results?.artifact_urls || {} : {};
    const viewerUrl = revisionUrls.noesis_aligned_glb
      || revisionUrls.merged_reconstruction_glb
      || pcfUrls.pcf_glb
      || (scan.alignment?.status === "complete"
        ? scan.alignment?.results?.artifact_urls?.comparison_glb
        : scan.outputs?.artifact_urls?.reconstruction_glb);
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

function uploadSupplementVideo(scan, file) {
  if (!file) return;
  if (uploadInProgress) {
    toastMessage("A video is already uploading");
    return;
  }
  uploadInProgress = true;
  uploadPanel.classList.remove("hidden");
  uploadLabel.textContent = `Adding ${file.name || "room video"} to ${scan.name}`;
  uploadPercent.textContent = "0%";
  uploadBar.style.width = "0%";
  toastMessage("Uploading and auto-saving the additional pass");
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/scans/${encodeURIComponent(scan.id)}/supplements`);
  xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
  xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name || "additional_walk.mp4"));
  xhr.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    uploadPercent.textContent = `${percent}%`;
    uploadBar.style.width = `${percent}%`;
  });
  xhr.addEventListener("load", async () => {
    uploadInProgress = false;
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = `${xhr.status} upload failed`;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) { /* retain HTTP detail */ }
      uploadLabel.textContent = detail;
      toastMessage(`Additional video was not saved: ${detail}`);
      return;
    }
    const updated = JSON.parse(xhr.responseText);
    const index = scans.findIndex((item) => item.id === scan.id);
    if (index >= 0) scans[index] = updated;
    uploadPercent.textContent = "100%";
    uploadBar.style.width = "100%";
    uploadLabel.textContent = "Additional video saved · preparing new views";
    lastDetailFingerprint = "";
    renderSelected();
    toastMessage("Additional pass auto-saved; frame preparation started");
    await refreshScans({ force: true });
    setTimeout(() => uploadPanel.classList.add("hidden"), 1800);
  });
  xhr.addEventListener("error", () => {
    uploadInProgress = false;
    uploadLabel.textContent = "Upload connection failed";
    toastMessage("Additional video upload failed. Confirm the phone is still on home Wi-Fi.");
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
