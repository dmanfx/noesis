import BrowserCapture from "./browser_capture.js?v=2.1.0";
import CompanionCapture from "./companion_capture.js?v=1.0.1";
import NativeCaptureUI, { isRoomWalkAndroid, pathReferenceForScan, pathSelectionLabel } from "./native_capture.js?v=2.3.0";
import CalibrationUI, { scanCameraSelectionMarkup } from "./calibration.js?v=2.2.0";
import { createPathComparisonController } from "./path_comparison.js?v=1.0.0";

const healthPill = document.querySelector("#health-pill");
const scanList = document.querySelector("#scan-list");
const scanDetail = document.querySelector("#scan-detail");
const scanName = document.querySelector("#scan-name");
const recordPhoneWalkButton = document.querySelector("#record-phone-walk-button");
const existingInput = document.querySelector("#existing-video");
const sensorBundleInput = document.querySelector("#sensor-bundle");
const browserCapturePanel = document.querySelector("#browser-capture-panel");
const browserPreview = document.querySelector("#browser-preview");
const browserCloseButton = document.querySelector("#browser-close-button");
const browserCaptureStatus = document.querySelector("#browser-capture-status");
const browserCaptureError = document.querySelector("#browser-capture-error");
const browserStartButton = document.querySelector("#browser-start-button");
const browserStopButton = document.querySelector("#browser-stop-button");
const browserDownloadButton = document.querySelector("#browser-download-button");
const browserRetryButton = document.querySelector("#browser-retry-button");
const browserDiscardButton = document.querySelector("#browser-discard-button");
const browserAccelerometerCount = document.querySelector("#browser-accelerometer-count");
const browserGyroscopeCount = document.querySelector("#browser-gyroscope-count");
const browserVideoFrameCount = document.querySelector("#browser-video-frame-count");
const browserCaptureBytes = document.querySelector("#browser-capture-bytes");
const companionCameraSelect = document.querySelector("#companion-camera-select");
const companionCameraStatus = document.querySelector("#companion-camera-status");
const companionCaptureStatus = document.querySelector("#companion-capture-status");
const companionHeartbeatStatus = document.querySelector("#companion-heartbeat-status");
const browserMarkerButton = document.querySelector("#browser-marker-button");
const browserMarkerRetryButton = document.querySelector("#browser-marker-retry-button");
const companionStopRetryButton = document.querySelector("#companion-stop-retry-button");
const uploadPanel = document.querySelector("#upload-panel");
const uploadLabel = document.querySelector("#upload-label");
const uploadPercent = document.querySelector("#upload-percent");
const uploadBar = document.querySelector("#upload-bar");
const refreshButton = document.querySelector("#refresh-button");
const newWalkButton = document.querySelector("#new-walk-button");
const captureCard = document.querySelector(".capture-card");
const workspace = document.querySelector(".workspace");
const toast = document.querySelector("#toast");
const secureCaptureNote = document.querySelector("#secure-capture-note");
const secureCaptureLink = document.querySelector("#secure-capture-link");
const secureCaptureCaLink = document.querySelector("#secure-capture-ca-link");
const captureModeReconstruction = document.querySelector("#capture-mode-reconstruction");
const captureModePath = document.querySelector("#capture-mode-path");
const reconstructionOptions = document.querySelector("#reconstruction-options");
const pathOptions = document.querySelector("#path-options");
const reconstructionTargetScan = document.querySelector("#reconstruction-target-scan");
const pathTargetScan = document.querySelector("#path-target-scan");
const pathReferenceStatus = document.querySelector("#path-reference-status");
const pathCameraSelect = document.querySelector("#path-camera-select");
const recordExistingVideoLabel = document.querySelector("#existing-video-label");
const sensorBundleLabel = document.querySelector("#sensor-bundle-label");
const pathBrowserNote = document.querySelector("#path-browser-note");

const NEW_WALK_MODE_KEY = "phoneScanNewWalkMode";
const CAPTURE_MODE_KEY = "phoneScanCaptureMode";
const RECONSTRUCTION_TARGET_KEY = "phoneScanReconstructionTarget";
const PATH_TARGET_KEY = "phoneScanPathTarget";
const PATH_CAMERA_KEY = "phoneScanPathCamera";
const WALK_INTENT_SCHEMA = "roomwalk.walk_intent.v1";
const LONG_PRESS_MS = 650;

let scans = [];
let newWalkMode = localStorage.getItem(NEW_WALK_MODE_KEY) === "1";
let selectedId = newWalkMode ? null : localStorage.getItem("phoneScanSelected") || null;
let captureMode = ["reconstruction", "path_refinement"].includes(localStorage.getItem(CAPTURE_MODE_KEY)) ? localStorage.getItem(CAPTURE_MODE_KEY) : "reconstruction";
let reconstructionTargetId = localStorage.getItem(RECONSTRUCTION_TARGET_KEY) || "";
let pathTargetId = localStorage.getItem(PATH_TARGET_KEY) || "";
let pathCameraId = localStorage.getItem(PATH_CAMERA_KEY) || "";
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
let pairedStaticAlignmentAvailable = false;
let browserBundleUploadFailed = false;
let activeBrowserIntent = null;
let activeWorkspaceTab = "capture";
let viewerModules = null;
let viewerGeneration = 0;
let serverUnavailable = false;
let calibrationUI = null;
const pathComparisonController = createPathComparisonController();

function handleCompanionFailure(error) {
  const message = error?.message || "The static room camera or tracking stream became unavailable.";
  toastMessage(`${message} The phone capture will stop and remain available for download or retry.`);
  if (browserCapture?.isRecording) void browserCapture.stop("companion_capture_lost").catch(() => {});
}

function normalizeCaptureMode(value) {
  return value === "path_refinement" ? "path_refinement" : "reconstruction";
}

function captureTargetId(mode = captureMode) {
  return mode === "path_refinement" ? pathTargetId : reconstructionTargetId;
}

function currentWalkIntent(mode = captureMode) {
  const normalized = normalizeCaptureMode(mode);
  const intent = {
    schema: WALK_INTENT_SCHEMA,
    mode: normalized,
    target_scan_id: captureTargetId(normalized) || null,
    carry_protocol: normalized === "path_refinement" ? "close_body" : "coverage",
    accuracy_target_m: 0.1,
  };
  if (normalized === "path_refinement") {
    const target = scans.find((scan) => scan?.id === pathTargetId);
    const reference = pathReferenceForScan(target);
    if (reference.status === "available") intent.target_reference = reference.selection;
  }
  return intent;
}

function targetScanLabel(scan, { path = false } = {}) {
  const base = `${scan.name || scan.id} · ${scan.status || "unknown"}`;
  if (!path) return base;
  return `${base} · ${pathReferenceForScan(scan).label}`;
}

function populateIntentTargets() {
  const completed = scans.filter((scan) => scan?.id && scan.status === "complete"
    && scan.outputs && typeof scan.outputs === "object" && !Array.isArray(scan.outputs)
    && (scan.walk_intent || scan.capture?.walk_intent)?.mode !== "path_refinement"
    && !scan.capture?.calibration_request && !scan.calibration_request);
  const optionMarkup = (selected, emptyLabel, { path = false } = {}) => `<option value="">${escapeHtml(emptyLabel)}</option>${completed.map((scan) => {
    const reference = path ? pathReferenceForScan(scan) : null;
    const disabled = reference?.status === "unavailable" ? " disabled" : "";
    const title = reference?.status === "unavailable" ? ` title="${escapeHtml(reference.reason)}"` : "";
    return `<option value="${escapeHtml(scan.id)}" ${scan.id === selected ? "selected" : ""}${disabled}${title}>${escapeHtml(targetScanLabel(scan, { path }))}</option>`;
  }).join("")}`;
  if (reconstructionTargetScan) {
    if (!completed.some((scan) => scan.id === reconstructionTargetId)) reconstructionTargetId = "";
    reconstructionTargetScan.innerHTML = optionMarkup(reconstructionTargetId, "New room reconstruction");
    reconstructionTargetScan.value = reconstructionTargetId;
  }
  if (pathTargetScan) {
    if (!completed.some((scan) => scan.id === pathTargetId)) pathTargetId = "";
    pathTargetScan.innerHTML = optionMarkup(pathTargetId, "Choose a completed reconstruction…", { path: true });
    pathTargetScan.value = pathTargetId;
  }
  if (pathCameraSelect) {
    const available = (nativeCaptureUI?.available ? nativeCaptureUI.snapshot?.room_cameras || [] : alignmentTargets).filter((target) => target?.camera_id);
    if (!available.some((target) => target.camera_id === pathCameraId)) pathCameraId = "";
    pathCameraSelect.innerHTML = `<option value="">Choose a paired camera…</option>${available.map((target) => `<option value="${escapeHtml(target.camera_id)}" ${target.camera_id === pathCameraId ? "selected" : ""}>${escapeHtml(target.label || target.camera_id)} camera</option>`).join("")}`;
    pathCameraSelect.value = pathCameraId;
  }
}

function captureIntentReady({ browser = false } = {}) {
  if (captureMode !== "path_refinement") return true;
  if (!pathTargetId) { toastMessage("Choose the existing reconstruction for this path refinement first."); return false; }
  const target = scans.find((scan) => scan?.id === pathTargetId);
  const reference = pathReferenceForScan(target);
  if (reference.status === "unavailable") {
    toastMessage(`${reference.label}. Choose a reconstruction with an available retained PCF reference or an explicit ordinary base.`);
    return false;
  }
  if (!pathCameraId) { toastMessage("Choose the paired Noesis camera before starting path refinement."); return false; }
  if (reference.status === "available" && reference.selection.camera_id !== pathCameraId) {
    toastMessage(`The selected PCF is bound to camera ${reference.selection.camera_id}, but path capture is set to ${pathCameraId}. Choose the bound camera before starting.`);
    return false;
  }
  if (browser) { toastMessage("Path refinement needs the native paired recorder. Browser video/import is reconstruction-only."); return false; }
  const snapshot = nativeCaptureUI.snapshot;
  if (snapshot?.room_cameras?.[snapshot.room_camera_index]?.camera_id !== pathCameraId) {
    toastMessage("The phone's room camera differs from the path selection. Select and apply the same camera in Connection and camera selection, then wait for the phone to confirm the selected paired camera.");
    return false;
  }
  return true;
}

const companionCapture = new CompanionCapture({
  onStateChange: renderCompanionCaptureState,
  onFailure: handleCompanionFailure,
});

const browserCapture = new BrowserCapture({
  onStateChange: renderBrowserCaptureState,
  onBundleReady: handleBrowserBundleReady,
  onBeforeFinalize: async ({ reason }) => {
    if (companionCapture.needsServerStop) {
      try {
        await companionCapture.stop(reason || "user");
      } catch (error) {
        // The phone TAR remains authoritative local evidence even when the
        // static stop request fails or times out.
        toastMessage(`Static capture stop failed: ${error.message || error}. Phone capture was preserved.`);
      }
    }
    browserCapture.setCompanionCapture(companionCapture.companionContext());
  },
});

const nativeCaptureUI = new NativeCaptureUI({
  captureRoot: document.querySelector("#native-capture-panel"),
  libraryRoot: document.querySelector("#native-library-panel"),
  platform: window,
  onError: (error) => toastMessage(error.message || String(error)),
  canCapture: (mode) => {
    if (uploadInProgress) { toastMessage("Finish the browser upload before starting native capture."); return false; }
    if (browserCaptureBlocked("start native capture")) return false;
    if (!["reconstruction", "path_refinement"].includes(mode)) return true;
    if (!captureIntentReady()) return false;
    const intent = currentWalkIntent(mode);
    if (intent.target_reference && nativeCaptureUI?.snapshot?.supports_target_reference !== true) {
      toastMessage("This RoomWalk app cannot preserve the selected PCF reference. Update the companion before recording this path.");
      return false;
    }
    return true;
  },
  onState: (snapshot) => { calibrationUI?.updateNative(snapshot); syncCaptureIntentUi(); },
  onUploaded: (scanId) => { void handleNativeUpload(scanId); },
  onPackaged: (snapshot) => {
    if (calibrationUI?.handlePackaged(snapshot)) return;
    activateWorkspaceTab("library");
    document.querySelector("#native-selected-card")?.scrollIntoView({ block: "start", behavior: "smooth" });
    toastMessage("Recording saved on this phone. Upload the selected capture when ready.");
  },
  onOpenLibrary: () => activateWorkspaceTab("library"),
});
calibrationUI = new CalibrationUI({
  root: document.querySelector("#calibration-panel"),
  nativeCapture: nativeCaptureUI,
  platform: window,
  onOpenScan: openScan,
});

function activateWorkspaceTab(name) {
  if (!["capture", "library", "setup"].includes(name)) return;
  if (name !== "capture" && (browserCapture.isRecording || browserCapture.status.state === "stopping")) {
    toastMessage("Stop and save the browser recording before leaving Capture.");
    return;
  }
  activeWorkspaceTab = name;
  for (const button of document.querySelectorAll("[data-workspace-tab]")) {
    const selected = button.dataset.workspaceTab === name;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    document.querySelector(`#panel-${button.dataset.workspaceTab}`)?.classList.toggle("hidden", !selected);
  }
  calibrationUI?.setVisible(name === "setup");
  disposeViewer();
  if (name === "library") { lastDetailFingerprint = ""; renderSelected(); }
}

async function handleNativeUpload(scanId) {
  // The native upload service retains its last receipt across new recordings.
  // A replay of that receipt must not move or interrupt the active capture UI.
  const native = nativeCaptureUI.snapshot;
  if (native?.active || native?.imu_active || native?.imu_preparing) return;
  try {
    // Fetch the exact receipt identity; an in-flight list refresh may predate
    // upload acceptance. Never select whichever walk happens to be newest.
    const scan = await jsonFetch(`/api/scans/${encodeURIComponent(scanId)}`);
    if (scan?.id !== scanId) throw new Error("The upload receipt does not match the returned walk.");
    const index = scans.findIndex((item) => item.id === scanId);
    if (index < 0) scans.unshift(scan); else scans[index] = scan;
    if (!calibrationUI.visible || !calibrationUI.workflow) openScan(scanId);
    calibrationUI.setScans(scans, { preferredId: scanId });
    if (calibrationUI.hasNativeCalibrationRequest || scan.capture?.calibration_request) activateWorkspaceTab("setup");
    else activateWorkspaceTab("library");
    const intent = scan.walk_intent;
    toastMessage(intent?.mode === "path_refinement"
      ? "Path refinement received. Review the retained path evidence when frame preparation is ready."
      : intent?.target_scan_id
        ? "Reconstruction capture received. Open it in Library to explicitly add its views to the linked reconstruction."
        : "Capture received by Noesis. Follow server validation before processing or reconstruction.");
  } catch (error) {
    toastMessage(`The phone retains the upload receipt. Refresh Library when the server is reachable: ${error.message}`);
  }
}

const statusLabels = {
  uploading: "Uploading",
  importing_capture: "Validating received capture",
  import_failed: "Capture validation needs attention",
  calibration_ready: "Calibration capture ready",
  processing_frames: "Preparing frames",
  ready: "Ready for inference",
  frame_failed: "Frame preparation failed",
  ma_queued: "MA queued",
  ma_running: "MapAnything running",
  ma_failed: "MapAnything failed",
  da3_queued: "DA3 queued",
  da3_running: "DA3 running",
  da3_failed: "DA3 failed",
  vio_queued: "OpenVINS queued",
  vio_running: "OpenVINS running",
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

function companionCaptureFor(scan) {
  return scan?.companion_capture || scan?.capture?.companion_capture || null;
}

function pairedCameraIdFor(scan) {
  const companion = companionCaptureFor(scan);
  const sessionId = typeof companion?.session_id === "string" ? companion.session_id.trim() : "";
  const cameraId = typeof companion?.camera_id === "string" ? companion.camera_id.trim() : "";
  return sessionId && cameraId ? cameraId : "";
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

function alignmentTargetControls(scan, selectedCameraId, buttonLabel) {
  const pairedCameraId = pairedCameraIdFor(scan);
  const selectedId = pairedCameraId || selectedCameraId;
  const pairedCapabilityReady = !pairedCameraId || pairedStaticAlignmentAvailable;
  const selectedIsAvailable = pairedCapabilityReady && Boolean(alignmentTargetFor(selectedId));
  const targets = pairedCameraId
    ? alignmentTargets.filter((target) => target.camera_id === pairedCameraId)
    : alignmentTargets;
  const options = targets.length
    ? targets.map((target) => `
    <option value="${escapeHtml(target.camera_id)}" ${target.camera_id === selectedId ? "selected" : ""}>
      ${escapeHtml(target.label)} camera
    </option>`).join("")
    : pairedCameraId
      ? `<option value="" selected disabled>${escapeHtml(cameraLabel(pairedCameraId))} camera is unavailable in the active release</option>`
      : "";
  const releaseNote = alignmentReleaseId
    ? `Validated scene release: ${escapeHtml(alignmentReleaseId)}`
    : "No validated scene release is available";
  const sourceLabel = pairedCameraId ? "Paired static camera" : "Saved static camera for this room";
  const sourceNote = pairedCameraId
    ? !pairedStaticAlignmentAvailable
      ? "Paired alignment is being updated; wait for the updated service before aligning."
      : selectedIsAvailable
        ? "Uses the static video recorded alongside this walk."
        : `Paired session ${escapeHtml(companionCaptureFor(scan)?.session_id)} has no matching validated target.`
    : releaseNote;
  const chooseOption = pairedCameraId && !targets.length
    ? ""
    : `<option value="" ${selectedIsAvailable ? "" : "selected"}>Choose camera…</option>`;
  return `
    <div class="alignment-actions">
      <label class="provider-picker alignment-picker">
        <span>${sourceLabel}</span>
        <select id="alignment-target" ${targets.length ? "" : "disabled"} ${pairedCameraId ? "disabled" : ""}>
          ${chooseOption}
          ${options}
        </select>
        <small>${sourceNote}</small>
      </label>
      <button id="align-noesis" class="primary-button" type="button" ${selectedIsAvailable ? "" : "disabled"} data-paired-camera-id="${escapeHtml(pairedCameraId)}">${escapeHtml(buttonLabel)}</button>
    </div>`;
}

function staticReferencePanel(scan) {
  const reference = scan?.alignment?.static_reference || scan?.static_reference || null;
  if (!reference || typeof reference !== "object") return "";
  const rawStatus = typeof reference.status === "string" ? reference.status.toLowerCase() : "unknown";
  const statusLabel = {
    building: "Building",
    complete: "Ready",
    failed: "Failed",
  }[rawStatus] || rawStatus;
  const message = typeof reference.message === "string" ? reference.message : "";
  const error = typeof reference.error === "string" ? reference.error : "";
  const artifactUrls = reference.artifact_urls && typeof reference.artifact_urls === "object"
    ? reference.artifact_urls
    : {};
  const artifactLabels = {
    keyframe: "Static keyframe",
    static_keyframe: "Static keyframe",
    depth_preview: "Static depth preview",
    manifest: "Static reference manifest",
    points: "Static reference points",
    static_points: "Static reference points",
  };
  const links = Object.entries(artifactUrls)
    .filter(([, url]) => typeof url === "string" && url)
    .map(([name, url]) => {
      const label = artifactLabels[name] || name.replaceAll("_", " ");
      return `<a class="artifact-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
    })
    .join("");
  const detail = error
    ? `<div class="error-box">${escapeHtml(error)}</div>`
    : message
      ? `<p class="alignment-note">${escapeHtml(message)}</p>`
      : "";
  return `<div class="status-panel alignment-static-reference"><div class="status-line"><span>Static reference build</span><b>${escapeHtml(statusLabel)}</b></div>${detail}${links ? `<div class="artifact-row">${links}</div>` : ""}</div>`;
}

function toastMessage(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3600);
}

function companionHeartbeatAge(status) {
  const heartbeat = status?.lastHeartbeat;
  if (!heartbeat) return "no heartbeat yet";
  const current = performance.now?.() ?? 0;
  const age = Math.max(0, current - Number(heartbeat.client_monotonic_ms || current));
  return `${(age / 1000).toFixed(1)}s ago`;
}

function renderCompanionCaptureState(status = companionCapture.status) {
  if (!companionCameraSelect) return;
  const pairedRequired = captureMode === "path_refinement";
  const cameras = Array.isArray(status?.cameras) ? status.cameras : [];
  const selected = status?.selectedCameraId || "";
  const active = ["starting", "recording", "stopping", "finalizing"].includes(status?.state);
  const options = ['<option value="">Choose the room camera…</option>']
    .concat(cameras.map((camera) => {
      const label = camera.label || camera.camera_id;
      const availability = camera.available === true ? "" : ` · unavailable${camera.reason ? `: ${camera.reason}` : ""}`;
      return `<option value="${escapeHtml(camera.camera_id)}" ${camera.camera_id === selected ? "selected" : ""} ${camera.available === true ? "" : "disabled"}>${escapeHtml(label)}${escapeHtml(availability)}</option>`;
    }));
  const currentOptions = companionCameraSelect.innerHTML;
  const nextOptions = options.join("");
  if (currentOptions !== nextOptions) companionCameraSelect.innerHTML = nextOptions;
  companionCameraSelect.value = selected;
  companionCameraSelect.disabled = active || !status?.camerasAvailable;

  const availableCount = cameras.filter((camera) => camera.available === true).length;
  if (!status?.camerasAvailable) {
    companionCameraStatus.textContent = pairedRequired
      ? status?.cameraReason || "No available static room cameras were reported by the appliance."
      : "Pairing is optional for reconstruction; no static room camera is selected.";
  } else if (active) {
    companionCameraStatus.textContent = `${availableCount} room camera${availableCount === 1 ? "" : "s"} available · selected ${status.cameraId || selected}`;
  } else if (selected) {
    companionCameraStatus.textContent = pairedRequired ? "Selected camera is required for path refinement." : "Selected camera will be retained as independent reconstruction reference.";
  } else {
    companionCameraStatus.textContent = pairedRequired ? "Choose the physically installed room camera before starting path refinement." : "Optional: choose a static room camera to retain paired reconstruction evidence.";
  }

  const staticStatus = status?.state === "recording"
    ? `Static camera recording · video ${status.videoStatus || "unknown"} · tracking ${status.trackingStatus || "unknown"}`
    : status?.state === "stopping" || status?.state === "finalizing"
      ? `Static capture ${status.state} · finalizing server recording`
      : status?.state === "stopped"
        ? "Static capture finalized"
        : status?.state === "failed"
          ? `Static capture failed${status.error ? `: ${status.error.message || status.error}` : ""}`
          : pairedRequired ? "Static capture is required before phone recording starts" : "Phone-only reconstruction can start without a static recording";
  if (companionCaptureStatus) companionCaptureStatus.textContent = staticStatus;
  if (companionHeartbeatStatus) {
    const heartbeat = status?.lastHeartbeat;
    const sequence = heartbeat?.tracking_sequence === null || heartbeat?.tracking_sequence === undefined ? "sequence unavailable" : `tracking sequence ${heartbeat.tracking_sequence}`;
    companionHeartbeatStatus.textContent = status?.state === "recording"
      ? `Heartbeat ${companionHeartbeatAge(status)} · ${sequence}`
      : status?.clockProbeCount ? `${status.clockProbeCount} server clock probes retained` : "No static session yet";
  }
  if (browserMarkerButton) {
    browserMarkerButton.disabled = status?.state !== "recording";
    browserMarkerButton.classList.toggle("hidden", status?.state !== "recording");
  }
  if (browserMarkerRetryButton) {
    browserMarkerRetryButton.disabled = status?.pendingMarkerCount <= 0 || status?.state !== "recording";
    browserMarkerRetryButton.classList.toggle("hidden", status?.pendingMarkerCount <= 0);
  }
  if (companionStopRetryButton) {
    companionStopRetryButton.disabled = status?.state === "recording" || !status?.needsServerStop;
    companionStopRetryButton.classList.toggle("hidden", !status?.needsServerStop || status?.state === "recording");
  }
  if (browserStartButton) {
    const phoneReady = browserCapture.status.state === "ready" && browserCapture.status.ready === true;
    browserStartButton.disabled = !phoneReady || (captureMode === "path_refinement" && !companionCapture.readyToStart);
  }
}

function renderBrowserCaptureState(status) {
  if (!browserCapturePanel) return;
  const state = status?.state || "idle";
  const counts = status?.counts || {};
  const error = status?.error;
  browserAccelerometerCount.textContent = Number(counts.accelerometer || 0).toLocaleString();
  browserGyroscopeCount.textContent = Number(counts.gyroscope || 0).toLocaleString();
  browserVideoFrameCount.textContent = Number(status?.videoFrames || 0).toLocaleString();
  browserCaptureBytes.textContent = formatBytes(status?.bytes || 0);
  const cameraSettings = status?.camera?.settings;
  const modeStatus = document.querySelector("#browser-mode-status");
  if (modeStatus) modeStatus.textContent = `${cameraSettings ? `Camera reports ${cameraSettings.width} × ${cameraSettings.height}. Encoded dimensions will be checked on upload.` : "Requests unscaled rear-camera 8K; unsupported modes stop setup."} Synchronization is unverified: this browser recorder cannot associate IMU acquisition times with encoded frames. The bounded capture holds about ${status?.estimatedMaxVideoSeconds || 31}s at the requested bitrate; actual capacity varies.`;
  renderCompanionCaptureState(companionCapture.status);
  browserStartButton.disabled = state !== "ready" || status.ready !== true || (captureMode === "path_refinement" && !companionCapture.readyToStart);
  browserStopButton.disabled = state !== "recording";
  browserCloseButton.disabled = ["recording", "stopping"].includes(state) || companionCapture.needsServerStop;
  browserDownloadButton.classList.toggle("hidden", !status.bundle);
  browserRetryButton.classList.toggle("hidden", !status.bundle || !browserBundleUploadFailed);
  browserDiscardButton.classList.toggle("hidden", !status.bundle);
  const modeLabel = status.sensorMode === "devicemotion_fallback"
    ? "DeviceMotion fallback (event timestamps, rotation converted from deg/s)"
    : status.sensorMode === "generic_sensor" ? "Generic Sensor API" : "sensor setup";
  const messages = {
    idle: "Camera and motion permission are required.",
    starting: "Requesting camera and motion access…",
    waiting_sensors: "Keep the phone stationary while both sensor streams initialize.",
    ready: `8K camera preview and both sensor streams are progressing (${modeLabel}). ${captureMode === "path_refinement" ? "Choose the paired static room camera." : "A static room camera is optional for reconstruction."} This records video and motion observations; synchronized capture requires the native recorder.`,
    recording: `Recording phone camera + motion with the selected static room capture (${modeLabel}). Walk slowly; ${Math.round(Number(status.durationMs || 0) / 1000)}s elapsed.`,
    stopping: "Stopping the phone recorder and finalizing the static capture…",
    complete: browserBundleUploadFailed ? "Capture is still in this tab. Retry upload or download before closing." : "Capture is ready. Video and motion are saved together. Timing and calibration still need checking before IMU-based reconstruction.",
    closed: "Browser capture is closed.",
    error: browserBundleUploadFailed && status.bundle ? "Capture is still in this tab. Retry upload or download before closing." : error?.message || "Browser capture stopped with an error. Any partial capture remains available for download or retry.",
  };
  browserCaptureStatus.textContent = messages[state] || "Browser capture ready.";
  browserCaptureError.classList.toggle("hidden", !error);
  if (error) {
    const help = error.helpUrl ? ` <a href="${escapeHtml(error.helpUrl)}" target="_blank" rel="noopener">secure-context guidance</a>` : "";
    browserCaptureError.innerHTML = `${escapeHtml(error.message || String(error))}${help}`;
  } else {
    browserCaptureError.textContent = "";
  }
}

function browserCaptureBlocked(action = "start another upload") {
  if (!browserCapture.hasUnsavedWork() && !companionCapture.hasUnsavedWork()) return false;
  browserCapturePanel?.classList.remove("hidden");
  toastMessage(`Finish or retry the current browser capture before you ${action}. Download it if you need a local copy.`);
  browserCapturePanel?.scrollIntoView({ behavior: "smooth", block: "center" });
  return true;
}

function handleBrowserBundleReady(bundle, status) {
  browserBundleUploadFailed = false;
  renderBrowserCaptureState(status);
  if (!bundle?.uploadable) {
    browserBundleUploadFailed = true;
    renderBrowserCaptureState(browserCapture.status);
    toastMessage("The browser bundle is available for download, but its byte limit prevents upload.");
    return;
  }
  uploadSensorBundle(bundle.blob, { fileName: bundle.fileName, fromBrowser: true });
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

function syncCaptureIntentUi() {
  captureMode = normalizeCaptureMode(captureMode);
  if (captureModeReconstruction) captureModeReconstruction.checked = captureMode === "reconstruction";
  if (captureModePath) captureModePath.checked = captureMode === "path_refinement";
  reconstructionOptions?.classList.toggle("hidden", captureMode !== "reconstruction");
  pathOptions?.classList.toggle("hidden", captureMode !== "path_refinement");
  captureCard?.classList.toggle("path-refinement-mode", captureMode === "path_refinement");
  const native = nativeCaptureUI?.available === true;
  const path = captureMode === "path_refinement";
  if (recordPhoneWalkButton) {
    recordPhoneWalkButton.textContent = path
      ? (native ? "Open paired path capture" : "Native app required for path refinement")
      : (native ? "Open native reconstruction capture" : "Record reconstruction walk");
    recordPhoneWalkButton.disabled = path && !native;
  }
  if (recordExistingVideoLabel) {
    recordExistingVideoLabel.textContent = path ? "Video import unavailable for path mode" : "Use video for reconstruction";
    recordExistingVideoLabel.classList.toggle("disabled-control", path);
  }
  if (sensorBundleLabel) {
    sensorBundleLabel.textContent = path ? "Sensor import unavailable for path mode" : "Import camera + IMU evidence";
    sensorBundleLabel.classList.toggle("disabled-control", path);
  }
  if (existingInput) existingInput.disabled = path;
  if (sensorBundleInput) sensorBundleInput.disabled = path;
  if (pathBrowserNote) pathBrowserNote.classList.toggle("hidden", !path || native);
  populateIntentTargets();
  const target = scans.find((scan) => scan?.id === pathTargetId);
  const reference = pathReferenceForScan(target);
  if (pathReferenceStatus) {
    pathReferenceStatus.classList.toggle("intent-warning", path && reference.status === "unavailable");
    pathReferenceStatus.textContent = !path || !pathTargetId
      ? "The selected reconstruction's retained reference will be used here; absence means the original reconstruction base."
      : reference.status === "available"
        ? `Selected path reference: ${reference.label}. The backend will verify its binding during path review.`
        : reference.status === "unavailable"
          ? `Selected path reference unavailable: ${reference.reason}. This path cannot silently use the ordinary base.`
          : `Selected path reference: ${reference.label}.`;
  }
  // An unavailable retained reference is a hard UI state, not an instruction to
  // send a base-mode intent. Keep the existing native intent visible but block
  // captureIntentReady() until the operator chooses a valid target.
  if (!(path && reference.status === "unavailable")) nativeCaptureUI?.setCaptureIntent?.(currentWalkIntent());
}

function setCaptureIntentMode(mode) {
  if (browserCapture?.hasUnsavedWork?.()) {
    toastMessage("Finish or discard the current browser capture before changing its purpose.");
    syncCaptureIntentUi();
    return;
  }
  captureMode = normalizeCaptureMode(mode);
  localStorage.setItem(CAPTURE_MODE_KEY, captureMode);
  syncCaptureIntentUi();
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
  calibrationUI?.setScans(scans, { preferredId: scanId });
  activateWorkspaceTab("library");
}

function startNewWalk() {
  if (uploadInProgress) {
    toastMessage("The current walk is still uploading. It will open as soon as it is safely saved.");
    return;
  }
  if (browserCaptureBlocked("start a new walk")) return;
  if (nativeCaptureUI.hasActiveWork()) {
    toastMessage("Finish the active native capture or transfer before starting a new walk.");
    return;
  }
  activateWorkspaceTab("capture");
  if (["starting", "waiting_sensors", "ready", "complete", "error"].includes(browserCapture.status.state)) {
    void browserCapture.close();
    browserCapturePanel.classList.add("hidden");
  }
  if (["failed", "stopped", "error"].includes(companionCapture.status.state) && !companionCapture.needsServerStop) companionCapture.reset();
  newWalkMode = true;
  selectedId = null;
  captureMode = "reconstruction";
  reconstructionTargetId = "";
  pathTargetId = "";
  pathCameraId = "";
  localStorage.setItem(CAPTURE_MODE_KEY, captureMode);
  localStorage.removeItem(RECONSTRUCTION_TARGET_KEY);
  localStorage.removeItem(PATH_TARGET_KEY);
  localStorage.removeItem(PATH_CAMERA_KEY);
  localStorage.setItem(NEW_WALK_MODE_KEY, "1");
  localStorage.removeItem("phoneScanSelected");
  scanName.value = "Phone room walk";
  existingInput.value = "";
  uploadPanel.classList.add("hidden");
  outputPage = 0;
  lastDetailFingerprint = "";
  disposeViewer();
  syncCaptureModeUi();
  renderScanList();
  syncCaptureIntentUi();
  renderSelected();
  (nativeCaptureUI.available ? document.querySelector("#native-capture-panel") : captureCard).scrollIntoView({ behavior: "smooth", block: "start" });
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

async function persistWalkIntent(scanId, intent) {
  if (!scanId || !intent) return null;
  try {
    return await jsonFetch(`/api/scans/${encodeURIComponent(scanId)}/walk-intent`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(intent),
    });
  } catch (error) {
    // Browser/imported evidence remains usable when the optional metadata route
    // has not reached the current service yet. Do not discard the source upload.
    toastMessage(`Capture saved, but its walk intent could not be recorded yet: ${error.message}`);
    return null;
  }
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
    calibrationUI?.setCapabilities(health.calibration);
    healthPill.textContent = `${health.device} · adaptive views · ${health.max_selected_frames} emergency max`;
    healthPill.className = "health-pill online";
    const calibrationNote = document.querySelector("#phone-calibration-note");
    const phoneCalibration = health.phone_camera_calibration;
    if (calibrationNote) {
      calibrationNote.classList.toggle("hidden", !phoneCalibration);
      if (phoneCalibration) {
        const resolution = (phoneCalibration.resolution || []).join(" × ");
        const mode = { uploaded_video: "uploaded videos", browser: "browser recordings", native_sensor_bundle: "native camera recordings" }[phoneCalibration.capture_mode];
        calibrationNote.textContent = mode
          ? `Phone calibration is available for ${resolution} ${mode} made with the same lens and capture settings as the calibration video.`
          : `Phone calibration imported (${resolution}). Its recording mode must be matched before it can be applied.`;
      }
    }
    const secureUrl = typeof health.secure_capture_url === "string" ? health.secure_capture_url : "";
    const caUrl = typeof health.ca_certificate_url === "string" ? health.ca_certificate_url : "";
    const showSecureHint = location.protocol !== "https:" && Boolean(secureUrl);
    secureCaptureNote.classList.toggle("hidden", !showSecureHint);
    if (showSecureHint) {
      secureCaptureLink.href = secureUrl;
      secureCaptureLink.textContent = secureUrl;
      secureCaptureCaLink.classList.toggle("hidden", !caUrl);
      if (caUrl) secureCaptureCaLink.href = caUrl;
    }
    const nextTargets = Array.isArray(health.alignment_targets)
      ? health.alignment_targets.filter((target) => target?.camera_id && target?.revision_id)
      : [];
    const nextReleaseId = health.alignment_release_id || null;
    const nextPcfPausesAppliance = health.pcf_pauses_appliance === true;
    const nextPairedStaticAlignmentAvailable = health.paired_static_alignment_available === true;
    const alignmentConfigChanged = JSON.stringify([
      alignmentTargets,
      alignmentReleaseId,
      pcfPausesAppliance,
      pairedStaticAlignmentAvailable,
    ]) !== JSON.stringify([
      nextTargets,
      nextReleaseId,
      nextPcfPausesAppliance,
      nextPairedStaticAlignmentAvailable,
    ]);
    alignmentTargets = nextTargets;
    alignmentReleaseId = nextReleaseId;
    pcfPausesAppliance = nextPcfPausesAppliance;
    pairedStaticAlignmentAvailable = nextPairedStaticAlignmentAvailable;
    populateIntentTargets();
    if (alignmentConfigChanged) {
      lastDetailFingerprint = "";
      renderSelected();
    }
  } catch (error) {
    healthPill.textContent = "Tool offline";
    calibrationUI?.setCapabilities(null, { offline: true });
    healthPill.className = "health-pill offline";
    if (pairedStaticAlignmentAvailable) {
      pairedStaticAlignmentAvailable = false;
      lastDetailFingerprint = "";
      renderSelected();
    }
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
  const importing = scan.status === "importing_capture";
  const running = ["uploading", "importing_capture", "processing_frames", "ma_queued", "ma_running", "da3_queued", "da3_running"].includes(scan.status);
  return `
    <div class="status-panel">
      <div class="status-line">
        <span>${escapeHtml(scan.message || statusLabels[scan.status] || scan.status)}</span>
        <b>${importing ? (scan.upload_import?.phase === "queued" ? "Queued" : "Validating") : running ? `${progress}%` : escapeHtml(statusLabels[scan.status] || scan.status)}</b>
      </div>
      ${running && !importing ? `<div class="progress-track"><span style="width:${progress}%"></span></div>` : ""}
      ${scan.error ? `<div class="error-box">${escapeHtml(scan.error)}</div>` : ""}
    </div>`;
}

function preparedSection(scan) {
  const prepared = scan.prepared;
  if (!prepared) return "";
  const selection = prepared.selection || {};
  const warnings = prepared.quality_warning_counts || {};
  const calibration = prepared.camera_calibration;
  const calibrationMessage = !calibration ? ""
    : calibration.status === "applied" ? "Phone camera calibration applied; lens distortion corrected."
    : calibration.reason_codes?.includes("capture_mode_not_bound") ? "Phone calibration is registered; its recording mode has not been matched yet."
    : "Phone calibration was not applied: this recording does not match its capture mode, dimensions, or orientation.";
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
    ${calibrationMessage ? `<p class="microcopy">${escapeHtml(calibrationMessage)}</p>` : ""}
    ${selection.selection_limited ? '<div class="warning-row"><span class="warning-chip">Emergency view ceiling reached</span></div>' : ""}
    ${warningHtml ? `<div class="warning-row">${warningHtml}</div>` : ""}
    <div class="media-panel">
      <img src="${prepared.contact_sheet_url}" alt="Prepared frame contact sheet" loading="lazy" />
      <div class="media-caption"><span>Prepared view coverage</span><a href="${prepared.manifest_url}" target="_blank" rel="noopener">Frame manifest</a></div>
    </div>
    <div class="subheading"><div><span class="eyebrow">Prepared input</span><h3>Adaptive reconstruction views</h3></div><span class="status-badge">First ${Math.min(24, prepared.frame_count)} shown</span></div>
    <div class="thumb-grid">${thumbs}</div>`;
}

function companionCaptureSection(companion) {
  if (!companion) return "";
  const links = Object.entries(companion.artifact_urls || {})
    .filter(([, url]) => typeof url === "string" && url)
    .slice(0, 12)
    .map(([name, url]) => `<a class="artifact-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(name.replaceAll("_", " "))}</a>`)
    .join("");
  return `<div class="companion-result"><div><span class="eyebrow">Paired static room recording</span><p><b>${escapeHtml(companion.camera_id || "Selected room camera")}</b> · ${escapeHtml(companion.status || "unknown")}</p><p>Retained with this phone walk for independent room reconstruction and alignment.</p><div class="sensor-checks"><span>video: ${escapeHtml(companion.video_status || companion.recorder?.status || "unknown")}</span><span>tracking: ${escapeHtml(companion.tracking_status || companion.tracking?.status || "unknown")}</span><span>${Number(companion.clock_probes?.length || 0)} clock probes</span></div>${companion.error ? `<p class="error-box">${escapeHtml(companion.error)}</p>` : ""}</div>${links ? `<div class="artifact-row">${links}</div>` : ""}</div>`;
}

function walkIntentReferenceLabel(intent) {
  if (intent?.mode !== "path_refinement") return "";
  if (!Object.prototype.hasOwnProperty.call(intent, "target_reference")) return "Original reconstruction · no retained PCF reference";
  try { return pathSelectionLabel(intent.target_reference); }
  catch (_) { return "Noesis PCF selection needs review · no fallback applied"; }
}

function walkIntentSection(scan) {
  const intent = scan?.walk_intent;
  if (!intent || typeof intent !== "object") {
    return `<section class="walk-intent-card legacy-intent"><div><span class="eyebrow">Capture intent</span><h3>Older capture · intent not recorded</h3><p>Original video, prepared views, calibration and IMU evidence remain available. This page does not infer whether an older walk was intended for reconstruction or path refinement.</p></div></section>`;
  }
  const mode = intent.mode === "path_refinement" ? "Path refinement" : "Reconstruction";
  const carry = intent.carry_protocol === "close_body" ? "close-body" : "coverage";
  const targetId = typeof intent.target_scan_id === "string" ? intent.target_scan_id : "";
  const target = scans.find((candidate) => candidate.id === targetId);
  const referenceLabel = walkIntentReferenceLabel(intent);
  const targetLink = target
    ? `<button class="ghost-button compact-button" type="button" data-open-target-scan="${escapeHtml(target.id)}">Open ${escapeHtml(target.name || target.id)}</button>`
    : targetId ? `<span class="status-badge">Target ${escapeHtml(targetId)} not in current library</span>` : "";
  const sourceReady = ["ready", "complete"].includes(scan.status);
  const targetReady = target?.status === "complete";
  const canAdd = mode === "Reconstruction" && target && target.id !== scan.id && sourceReady && targetReady;
  const addAction = canAdd ? `<button id="add-views-to-target" class="primary-button compact-button" type="button" data-target-scan="${escapeHtml(target.id)}">Add these views to target</button>` : "";
  const note = mode === "Path refinement"
    ? `Self-carried path: phone against your own torso, elbows tucked and stable, with whole-body turns and movement. Selected reference: ${referenceLabel}. Review compares the original visual path with retained IMU diagnostics and paired Noesis references; it does not prove 10 cm accuracy.`
    : target
      ? "This source capture stays retained separately. Adding its prepared views is an explicit, versioned action."
      : "RGB reconstruction is available without metric calibration; useful saved evidence remains downloadable below.";
  return `<section class="walk-intent-card ${mode === "Path refinement" ? "path-intent" : "reconstruction-intent"}">
    <div class="walk-intent-copy"><span class="eyebrow">Capture intent</span><h3>${mode}</h3><div class="intent-facts"><span>Protocol: ${escapeHtml(carry)}</span><span>Target: ${target ? escapeHtml(target.name || target.id) : "new room"}</span>${referenceLabel ? `<span>Reference: ${escapeHtml(referenceLabel)}</span>` : ""}<span>Desired validation target: ${Number(intent.accuracy_target_m) === 0.1 ? "10 cm" : "not reported"}</span></div><p>${escapeHtml(note)}</p></div>
    <div class="walk-intent-actions">${targetLink}${addAction}</div>
  </section>`;
}

function pathReviewSection(scan) {
  const intent = scan?.walk_intent || scan?.capture?.walk_intent;
  const legacyNative = !intent && scan?.capture?.capture_kind === "android_camera_imu"
    && !scan.capture.calibration_request && !scan.calibration_request && scan.status !== "calibration_ready";
  if (intent?.mode !== "path_refinement" && !legacyNative) return "";
  const protocolNote = legacyNative ? "<p class=\"microcopy\">Retained camera path; original carry protocol unknown. Pairing is shown only when recorded in the retained evidence.</p>" : "";
  const review = scan.path_review && typeof scan.path_review === "object" ? scan.path_review : null;
  const status = String(review?.status || "").toLowerCase();
  const progress = Math.round(Math.max(0, Math.min(1, Number(review?.progress || 0))) * 100);
  const results = review?.results && typeof review.results === "object" ? review.results : {};
  const reviewReference = results.reference_reconstruction && typeof results.reference_reconstruction === "object" ? results.reference_reconstruction : {};
  const referenceLabel = typeof reviewReference.label === "string" && reviewReference.label.trim()
    ? reviewReference.label.trim().slice(0, 240)
    : walkIntentReferenceLabel(intent);
  const referenceNote = referenceLabel ? `<p class="microcopy path-reference-identity">Selected path reference: ${escapeHtml(referenceLabel)}. The backend will verify the retained manifest and frame binding; no metric accuracy is implied.</p>` : "";
  const artifact = results.artifact_url || results.artifact_urls?.artifact || review?.artifact_url || "";
  const artifactLabels = { report: "Review report", trajectory: "Visual trajectory", motion: "IMU consistency", motion_intervals: "Motion intervals", reference_pcf_manifest: "Selected PCF manifest", reference_pcf_points: "Selected PCF points", reference_pcf_binding: "Selected PCF binding" };
  const artifactUrls = results.artifact_urls || review?.artifact_urls || {};
  const artifactLinks = Object.entries(artifactUrls)
    .filter(([name, url]) => /^[A-Za-z0-9_.-]{1,80}$/.test(name) && typeof url === "string" && (url.startsWith("/") || /^https?:\/\//i.test(url)))
    .slice(0, 32)
    .map(([name, url]) => `<a class="artifact-link" href="${escapeHtml(url)}" target="_blank" rel="noopener" download>${escapeHtml(artifactLabels[name] || name.replaceAll("_", " "))}</a>`)
    .join("");
  const comparison = results.path_comparison || review?.path_comparison || {};
  const refinement = results.path_refinement || review?.path_refinement || results.sensor_refined_path || {};
  const provider = String(results.provider || scan.outputs?.provider || scan.provider || "").toLowerCase();
  const modelNote = provider === "mapanything"
    ? "<p class=\"microcopy path-model-note\">Path model: MapAnything · comparison only. Run DA3 for a position-refined path when the backend reports one.</p>"
    : provider === "da3"
      ? "<p class=\"microcopy path-model-note\">Path model: DA3 · position refinement is shown only when the backend reports it.</p>"
      : "";
  const evidenceSummary = status === "complete" ? `<div class="path-review-evidence"><span>Visual path: ${escapeHtml(results.visual_path?.status || "not reported")} · ${Number(results.visual_path?.pose_count || 0).toLocaleString()} poses</span><span>Gyro diagnostic: ${escapeHtml(results.imu_consistency?.status || results.gyro_diagnostic?.status || "not reported")}</span><span>Body-ground reference: ${escapeHtml(results.body_ground_reference?.status || "not established")}</span></div><p class="microcopy">Desired target: ${Number(results.accuracy?.target_m ?? intent?.accuracy_target_m) === 0.1 ? "10 cm" : "not reported"}. Qualified: no. Measured position error: ${results.accuracy?.measured_position_error_m == null ? "not reported" : `${Number(results.accuracy.measured_position_error_m).toFixed(3)} m`}. Separation remains a review diagnostic, not an accuracy claim.</p>` : "";
  const canReview = scan.status === "complete" && scan.outputs && typeof scan.outputs === "object" && !Array.isArray(scan.outputs);
  if (["queued", "running"].includes(status)) return `<section class="path-review-card status-panel"><div class="status-line"><span>${escapeHtml(review.message || "Preparing retained path review")}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div><p class="alignment-target-note">Review-only: original visual path, IMU consistency, and any retained paired references.</p>${referenceNote}${protocolNote}</section>`;
  if (status === "complete") return `<section class="path-review-card ready-callout"><div><span class="eyebrow">Path review</span><h3>Review package ready</h3><p>${escapeHtml(review.message || "The original path and retained evidence are available for inspection. This is not 10 cm accuracy proof.")}</p>${referenceNote}${protocolNote}${modelNote}${evidenceSummary}<div id="path-comparison-host" class="path-comparison-host" data-path-comparison-scan="${escapeHtml(scan.id)}"></div><div class="artifact-row">${artifact ? `<a class="primary-button compact-button" href="${escapeHtml(artifact)}" target="_blank" rel="noopener">Export path review</a>` : ""}${artifactLinks}</div><button id="review-path" class="secondary-button compact-button" type="button" ${canReview ? "" : "disabled"}>${canReview ? "Rerun review" : "Review retained result"}</button></div></section>`;
  if (status === "failed") return `<section class="path-review-card status-panel"><div class="status-line"><span>Path review needs attention</span><b>Not completed</b></div><div class="error-box">${escapeHtml(review.error || "The retained path review could not be prepared.")}</div>${referenceNote}${protocolNote}<button id="review-path" class="secondary-button compact-button" type="button" ${canReview ? "" : "disabled"}>Retry path review</button></section>`;
  return `<section class="path-review-card ready-callout"><div><span class="eyebrow">Path review</span><h3>${legacyNative ? "Inspect the retained camera path" : "Inspect the paired walk"}</h3><p>${canReview ? "Export the original visual path, gyro diagnostic eligibility, and any retained paired references." : "Run reconstruction below to produce this walk’s visual camera trajectory before reviewing the path."} The 10 cm target is a validation target, not a proof claim.</p>${referenceNote}${protocolNote}${modelNote}</div><button id="review-path" class="primary-button compact-button" type="button" ${canReview ? "" : "disabled"}>Review path</button></section>`;
}

function captureSection(scan) {
  const capture = scan.capture || (scan.companion_capture ? { schema: "noesis.phone_capture.browser.v1", companion_capture: scan.companion_capture } : null);
  if (!capture) return "";
  const isBrowserCapture = capture.schema === "noesis.phone_capture.browser.v1"
    || capture.kind === "browser"
    || scan.input_mode === "browser_capture";
  if (isBrowserCapture) {
    const sensors = capture.sensors || {};
    const accel = sensors.accelerometer || {};
    const gyro = sensors.gyroscope || {};
    const count = (sensor) => Number(sensor.sample_count ?? sensor.samples?.length ?? 0).toLocaleString();
    const durationMs = Number(capture.timing?.duration_ms ?? 0);
    const stopReason = capture.stop_reason || scan.stop_reason || "unknown";
    const manifestUrl = capture.manifest_url || capture.urls?.manifest || scan.urls?.capture_manifest;
    const videoUrl = capture.video_url || capture.urls?.video || scan.video?.url;
    const links = [
      manifestUrl ? `<a class="artifact-link" href="${escapeHtml(manifestUrl)}" target="_blank" rel="noopener">Capture manifest</a>` : "",
      videoUrl ? `<a class="artifact-link" href="${escapeHtml(videoUrl)}" target="_blank" rel="noopener">Captured video</a>` : "",
    ].filter(Boolean).join("");
    const companionHtml = companionCaptureSection(capture.companion_capture || scan.companion_capture);
    return `<div class="ready-callout sensor-callout browser-saved-capture"><div><span class="eyebrow">Browser camera + IMU</span><h3>${escapeHtml(capture.device?.model || "Browser capture")}</h3><p>Video and motion were collected in this browser session. Timing and calibration still need checking before IMU-based reconstruction.</p><div class="sensor-checks"><span>${count(accel)} accelerometer samples</span><span>${count(gyro)} gyroscope samples</span><span>${formatDuration(durationMs / 1000)} captured</span><span>stop reason: ${escapeHtml(stopReason)}</span></div></div>${links ? `<div class="artifact-row">${links}</div>` : ""}${companionHtml}</div>`;
  }
  const vio = scan.vio || {};
  const metricReady = capture.metric_vio_allowed === true;
  const running = ["queued", "running"].includes(vio.status);
  const canRun = metricReady && scan.prepared && !running && [undefined, "failed"].includes(vio.status);
  const status = running
    ? `<div class="status-panel"><div class="status-line"><span>${escapeHtml(vio.message || "Running OpenVINS")}</span><b>${Math.round(Number(vio.progress || 0) * 100)}%</b></div><div class="progress-track"><span style="width:${Math.round(Number(vio.progress || 0) * 100)}%"></span></div></div>`
    : vio.status === "failed" ? `<div class="error-box">${escapeHtml(vio.error || "OpenVINS failed")}</div>` : "";
  const result = vio.status === "complete" && vio.results ? `<span class="status-badge aligned-badge">${Number(vio.results.poses?.length || 0)} camera poses · metric VIO</span>` : "";
  const motion = scan.prepared?.imu_motion;
  const motionText = motion?.status === "available"
    ? `IMU motion was used to score ${Number(motion.candidate_count_with_motion || 0)} candidate views. Exact frame timestamps, angular speed, and acceleration evidence are retained in the frame manifest.`
    : "Raw IMU streams are retained with the phone video for later analysis.";
  const links = [
    [scan.video?.url, "Phone video"], [capture.imu_normalized_url, "IMU samples"],
    [capture.video_timestamps_url, "Frame timestamps"], [capture.import_report_url, "Capture evidence"],
    [capture.manifest_url, "Capture manifest"], [scan.prepared?.manifest_url, "Frame motion evidence"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${escapeHtml(url)}" target="_blank" rel="noopener">${label}</a>`).join("");
  return `<div class="ready-callout sensor-callout"><div><span class="eyebrow">Native phone video + IMU</span><h3>${escapeHtml(capture.device?.model || "Camera + IMU bundle")}</h3><p>${escapeHtml(motionText)}</p><p>${metricReady ? "Calibration and coverage admit metric VIO." : "Metric camera poses require validated camera/IMU calibration; this does not prevent capture or reconstruction."}</p></div><div class="sensor-actions">${result}${canRun ? '<button id="initiate-vio" class="primary-button" type="button">Run OpenVINS</button>' : ""}</div>${links ? `<div class="artifact-row">${links}</div>` : ""}${companionCaptureSection(capture.companion_capture || scan.companion_capture)}</div>${status}`;
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
    const paired = Boolean(pairedCameraIdFor(scan));
    const heading = paired ? "Register this walk to its paired room camera." : "Register this walk to the correct room camera.";
    const note = paired
      ? pairedStaticAlignmentAvailable
        ? "Alignment will build a static reference from the camera video recorded alongside this walk."
        : "Paired alignment is being updated; wait for the updated service before starting."
      : "Choose the static camera physically installed in this room. The tool will preserve metric scale and gravity, fit against that camera's validated reconstruction, and reject ambiguous fits.";
    return `<div class="ready-callout alignment-callout"><div><h3>${heading}</h3><p>${note}</p></div>${alignmentTargetControls(scan, "", "Align to selected camera")}</div>${staticReferencePanel(scan)}`;
  }
  if (["queued", "running"].includes(alignment.status)) {
    const progress = Math.round(Math.max(0, Math.min(1, Number(alignment.progress || 0))) * 100);
    const source = alignment.target_kind === "paired_static" || pairedCameraIdFor(scan)
      ? "paired static recording"
      : "saved static reference";
    const revisionLabel = alignment.target_revision_id || "reference being prepared";
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Aligning to Noesis")}</span><b>${progress}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div><p class="alignment-target-note">Target: ${escapeHtml(cameraLabel(alignment.target_camera_id))} · ${escapeHtml(revisionLabel)} · ${source}</p>${staticReferencePanel(scan)}</div>`;
  }
  if (alignment.status === "failed") {
    return `<div class="status-panel alignment-status"><div class="status-line"><span>${escapeHtml(alignment.message || "Noesis alignment failed")}</span><b>Not aligned</b></div><div class="error-box">${escapeHtml(alignment.error || "The automatic fit did not pass its quality gate.")}</div><p class="alignment-target-note">Previous target: ${escapeHtml(cameraLabel(alignment.target_camera_id))}</p>${staticReferencePanel(scan)}${alignmentTargetControls(scan, alignment.target_camera_id || "", "Try selected camera")}</div>`;
  }
  const results = alignment.results;
  if (alignment.status !== "complete" || !results) return staticReferencePanel(scan);
  const urls = results.artifact_urls || {};
  const vertical = results.vertical_structure || {};
  const full = results.full_cloud || {};
  const reprojection = results.fixed_camera_reprojection || {};
  const paired = alignment.target_kind === "paired_static" || Boolean(pairedCameraIdFor(scan));
  const targetLabel = cameraLabel(results.target_camera_id || alignment.target_camera_id);
  const links = [
    [urls.aligned_phone_glb, "Aligned RGB GLB"],
    [urls.comparison_glb, "Noesis comparison GLB"],
    [urls.transform, "Phone → Noesis transform"],
    [urls.trajectory, "Aligned camera poses"],
    [urls.camera_solution_npz, "Aligned solution NPZ"],
    [urls.report, "Quality report"],
  ].filter(([url]) => url).map(([url, label]) => `<a class="artifact-link" href="${url}" target="_blank" rel="noopener">${label}</a>`).join("");
  return `${staticReferencePanel(scan)}
    <div class="subheading"><div><span class="eyebrow">Noesis registration</span><h3>${escapeHtml(targetLabel)} alignment passed</h3></div><span class="status-badge aligned-badge">Backend world · metric</span></div>
    <p class="alignment-note">This is a saved, quality-gated review candidate aligned against ${escapeHtml(results.target_revision_id || alignment.target_revision_id || "the selected revision")} using the ${paired ? "paired static recording" : "saved static reference"}. It has not changed or promoted the live Noesis world.</p>
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
  const pcfConfidence = results.pcf_confidence || scan.pcf_confidence || {};
  const metric = (value, suffix = "") => Number.isFinite(Number(value)) ? `${Number(value).toFixed(3)}${suffix}` : "—";
  const fraction = (value) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
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
        <div class="stat"><b>${fraction(heldout.odd_frame_valid_pixel_coverage_fraction)}</b><span>Internal comparison coverage</span></div>
        <div class="stat"><b>${metric(heldout.even_frame_map_to_odd_frame_depth_median_m, "m")}</b><span>Internal depth residual</span></div>
        <div class="stat"><b>${fraction(heldout.large_error_fraction_gt_2m ?? pcfConfidence.large_error_fraction_gt_2m)}</b><span>&gt;2m internal error</span></div>
        <div class="stat"><b>${heldout.independent === false ? "Internal only" : heldout.independent === true ? "Independent" : "—"}</b><span>Comparison provenance</span></div>
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
    ${scan.walk_intent?.mode === "path_refinement" ? "" : supplementsSection(scan)}
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
  viewerGeneration += 1;
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

async function initializeViewer(glbUrl) {
  const host = document.querySelector("#viewer-canvas");
  const overlay = document.querySelector("#viewer-overlay");
  if (!host || !glbUrl || activeWorkspaceTab !== "library") return;
  disposeViewer();
  const generation = viewerGeneration;
  let THREE, OrbitControls, GLTFLoader;
  try {
    // A failed/offline 3D dependency must never prevent capture, native bridge
    // registration, board editing, or the rest of the Library from booting.
    viewerModules ||= Promise.all([
      import("three"),
      import("/three/examples/controls/OrbitControls.js"),
      import("/three/examples/loaders/GLTFLoader.js"),
    ]);
    const modules = await viewerModules;
    [THREE, { OrbitControls }, { GLTFLoader }] = modules;
  } catch (error) {
    viewerModules = null;
    if (generation === viewerGeneration && host.isConnected) overlay.textContent = `3D review unavailable: ${error.message || "viewer dependencies could not load"}. Capture and saved evidence remain available.`;
    return;
  }
  if (generation !== viewerGeneration || !host.isConnected || activeWorkspaceTab !== "library") return;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0a);
  const camera = new THREE.PerspectiveCamera(52, 1, 0.001, 10000);
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }); }
  catch (error) { overlay.textContent = `3D rendering is unavailable: ${error.message || "WebGL unavailable"}. Saved artifact links remain usable.`; return; }
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
    if (generation !== viewerGeneration) return;
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
      if (generation !== viewerGeneration || !host.isConnected) {
        root.traverse((object) => {
          object.geometry?.dispose?.();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) material?.dispose?.();
        });
        return;
      }
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
      if (generation === viewerGeneration) overlay.textContent = `3D viewer could not load: ${error?.message || "unknown error"}`;
    },
  );
}

function wireDetailActions(scan) {
  document.querySelector("#open-calibration")?.addEventListener("click", () => {
    calibrationUI.setScans(scans, { preferredId: scan.id });
    activateWorkspaceTab("setup");
  });
  document.querySelector("#rename-scan")?.addEventListener("click", () => {
    void renameScan(scan.id);
  });
  document.querySelectorAll("[data-open-target-scan]").forEach((button) => {
    button.addEventListener("click", () => openScan(button.dataset.openTargetScan));
  });
  document.querySelector("#add-views-to-target")?.addEventListener("click", async (event) => {
    const targetId = event.currentTarget.dataset.targetScan;
    if (!targetId || !scan.walk_intent?.target_scan_id) return;
    event.currentTarget.disabled = true;
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(targetId)}/supplements/from-scan?source_scan_id=${encodeURIComponent(scan.id)}`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === targetId);
      if (index >= 0 && updated?.id === targetId) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage("These views were queued as an explicit additive revision; the source capture remains retained.");
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`Views were not added: ${error.message}`);
    }
  });
  document.querySelector("#review-path")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try {
      const targetId = typeof scan.walk_intent?.target_scan_id === "string" ? scan.walk_intent.target_scan_id : "";
      const targetQuery = targetId ? `?target_scan_id=${encodeURIComponent(targetId)}` : "";
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/path-review${targetQuery}`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage("Path review queued. The original visual, IMU, and paired evidence remains unchanged.");
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`Path review could not start: ${error.message}`);
    }
  });
  const pathComparisonHost = document.querySelector("#path-comparison-host");
  if (pathComparisonHost) {
    const review = scan.path_review && typeof scan.path_review === "object" ? scan.path_review : {};
    const results = review.results && typeof review.results === "object" ? review.results : {};
    void pathComparisonController.render({
      host: pathComparisonHost,
      scanId: scan.id,
      summary: results.path_comparison || review.path_comparison || {},
      pathRefinement: results.path_refinement || review.path_refinement || results.sensor_refined_path || {},
      artifactUrls: results.artifact_urls || review.artifact_urls || {},
    });
  } else {
    pathComparisonController.cancel();
  }
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
  document.querySelector("#initiate-vio")?.addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try {
      const updated = await jsonFetch(`/api/scans/${encodeURIComponent(scan.id)}/initiate-vio`, { method: "POST" });
      const index = scans.findIndex((item) => item.id === scan.id);
      if (index >= 0) scans[index] = updated;
      lastDetailFingerprint = "";
      renderSelected();
      toastMessage("OpenVINS started on the synchronized capture");
    } catch (error) {
      event.currentTarget.disabled = false;
      toastMessage(`OpenVINS could not start: ${error.message}`);
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
    if (!alignmentButton) return;
    const pairedCameraId = alignmentButton.dataset.pairedCameraId || "";
    alignmentButton.disabled = pairedCameraId
      ? alignmentTarget.value !== pairedCameraId || !alignmentTargetFor(pairedCameraId)
      : !alignmentTarget.value;
  });
  alignmentButton?.addEventListener("click", async (event) => {
    const cameraId = alignmentTarget?.value || "";
    const pairedCameraId = event.currentTarget.dataset.pairedCameraId || "";
    if (pairedCameraId && !pairedStaticAlignmentAvailable) {
      toastMessage("Paired alignment is being updated; wait for the updated service before aligning.");
      event.currentTarget.disabled = true;
      return;
    }
    if (pairedCameraId && cameraId !== pairedCameraId) {
      toastMessage(`This paired walk must use the ${cameraLabel(pairedCameraId)} static recording.`);
      event.currentTarget.disabled = true;
      return;
    }
    const target = alignmentTargetFor(cameraId);
    if (!target) {
      toastMessage(pairedCameraId
        ? `The paired ${cameraLabel(pairedCameraId)} camera has no matching validated target.`
        : "Choose the static camera for this room before aligning");
      event.currentTarget.disabled = true;
      return;
    }
    const accepted = window.confirm(
      pairedCameraId
        ? `Align “${scan.name}” using the paired ${target.label} static recording?`
        : `Align “${scan.name}” to the ${target.label} camera reconstruction?`,
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
    pathComparisonController.cancel();
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
  pathComparisonController.cancel();
  disposeViewer();
  const alignmentRunning = ["queued", "running"].includes(scan.alignment?.status);
  const supplementRunning = (scan.supplements || []).some((addition) => ["uploading", "processing_frames", "queued", "running"].includes(addition.status));
  const pcfRunning = ["queued", "running"].includes(scan.pcf?.status);
  const pathReviewRunning = ["queued", "running"].includes(scan.path_review?.status);
  const canDelete = !["uploading", "importing_capture", "processing_frames", "ma_queued", "ma_running", "da3_queued", "da3_running"].includes(scan.status) && !alignmentRunning && !supplementRunning && !pcfRunning && !pathReviewRunning;
  const canInitiate = ["ready", "ma_failed", "da3_failed"].includes(scan.status) && !pathReviewRunning;
  const pathScan = scan.walk_intent?.mode === "path_refinement";
  const requestedProvider = [scan.provider, scan.requested_provider, scan.selected_provider].find((value) => ["da3", "mapanything"].includes(value)) || "";
  const providerChoice = requestedProvider || (pathScan ? "da3" : "mapanything");
  const videoSize = formatBytes(scan.video?.size_bytes);
  scanDetail.innerHTML = `
    <div class="detail-header">
      <div class="detail-title">
        <span class="eyebrow">${escapeHtml(statusLabels[scan.status] || scan.status)}</span>
        <h2>${escapeHtml(scan.name)}</h2>
        <div class="detail-meta"><span>${escapeHtml(formatDate(scan.created_at))}</span><span>${videoSize}</span><span>${escapeHtml(scan.id)}</span></div>
      </div>
      <div class="detail-actions">
        ${scan.walk_intent?.mode !== "path_refinement" && scan.status === "complete" && !supplementRunning && !pcfRunning ? '<label class="secondary-button" for="additional-camera-video">＋ Add Video</label><input id="additional-camera-video" class="visually-hidden" type="file" accept="video/*" capture="environment" />' : ""}
        ${scan.video?.url ? `<a class="ghost-button" href="${scan.video.url}" target="_blank" rel="noopener">Original video</a>` : ""}
        ${scan.capture?.calibration_request || scan.status === "calibration_ready" ? '<button id="open-calibration" class="primary-button" type="button">Open Setup</button>' : ""}
        <button id="rename-scan" class="ghost-button" type="button">Rename</button>
        ${canDelete ? '<button id="delete-scan" class="danger-button" type="button">Delete scan</button>' : ""}
      </div>
    </div>
    ${statusPanel(scan)}
    ${walkIntentSection(scan)}
    ${pathReviewSection(scan)}
    ${scanCameraSelectionMarkup(scan)}
    ${captureSection(scan)}
    ${canInitiate ? `<div class="ready-callout inference-callout"><div><h3>${scan.status.endsWith("_failed") ? "The prepared views are still safe." : pathScan ? "The path walk is ready." : "The room walk is ready."}</h3><p>Reconstruction uses all ${Number(scan.prepared?.frame_count || 0)} adaptive phone views. ${pathScan ? "DA3 is the path model for position refinement when the backend accepts it; MapAnything remains available for comparison-only review." : "MapAnything automatically uses overlapping registered windows above its measured joint-view capacity. DA3 uses DA3-BASE any-view geometry plus the validated DA3Metric-Large FP16 TensorRT engine for metric scale."}</p></div><label class="provider-picker"><span>${pathScan ? "Path model" : "Provider"}</span><select id="inference-provider"><option value="da3" ${providerChoice === "da3" ? "selected" : ""}>DA3${pathScan ? " · position refinement" : ""}</option><option value="mapanything" ${providerChoice === "mapanything" ? "selected" : ""}>MapAnything${pathScan ? " · comparison only" : ""}</option></select>${pathScan ? '<small class="provider-note">Choosing MapAnything does not produce a position-refined path.</small>' : ""}</label><button id="initiate-inference" class="primary-button" type="button">${pathScan ? "Run path reconstruction" : "Run reconstruction"}</button></div>` : ""}
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
    serverUnavailable = false;
    populateIntentTargets();
    document.querySelector("#library-connection").textContent = "Server library · videos, prepared views, and reconstruction evidence retained on the Noesis machine.";
    calibrationUI?.setScans(scans);
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
    document.querySelector("#library-connection").textContent = "Server library unavailable. Previously displayed walks may be stale; native phone captures and board editing remain available offline.";
    if (force || !serverUnavailable) toastMessage(`Could not refresh scans: ${error.message}`);
    serverUnavailable = true;
  } finally {
    refreshing = false;
  }
}

function uploadVideo(file) {
  if (!file) return;
  if (!captureIntentReady({ browser: true })) return;
  if (browserCaptureBlocked("use another video")) return;
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
    existingInput.value = "";
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = `${xhr.status} upload failed`;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) { /* keep HTTP detail */ }
      uploadLabel.textContent = detail;
      toastMessage(detail);
      return;
    }
    let scan = JSON.parse(xhr.responseText);
    const intentResponse = await persistWalkIntent(scan.id, activeBrowserIntent || currentWalkIntent());
    if (intentResponse?.id === scan.id) scan = intentResponse;
    activeBrowserIntent = null;
    rememberSelectedScan(scan.id);
    uploadPercent.textContent = "100%";
    uploadBar.style.width = "100%";
    uploadLabel.textContent = "Upload saved · preparing multi-view frames";
    toastMessage("Walk auto-saved; frame preparation started");
    await refreshScans({ force: true });
    activateWorkspaceTab("library");
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

function uploadSensorBundle(file, { fileName = file?.name, fromBrowser = false } = {}) {
  if (!file) return;
  if (!captureIntentReady({ browser: true })) return;
  if (!fromBrowser && browserCaptureBlocked("import another sensor bundle")) return;
  if (uploadInProgress) {
    toastMessage("A walk is already uploading");
    return;
  }
  uploadInProgress = true;
  uploadPanel.classList.remove("hidden");
  const uploadName = fileName || file.name || "phone_capture.tar";
  const browserBundle = fromBrowser ? browserCapture.status.bundle : null;
  uploadLabel.textContent = `Uploading sensor bundle ${uploadName}`;
  uploadPercent.textContent = "0%";
  uploadBar.style.width = "0%";
  const xhr = new XMLHttpRequest();
  const name = scanName.value.trim() || "Phone room walk with sensors";
  xhr.open("POST", `/api/scans/sensor-bundle?name=${encodeURIComponent(name)}`);
  xhr.setRequestHeader("Content-Type", file.type || "application/x-tar");
  xhr.setRequestHeader("X-File-Name", encodeURIComponent(uploadName));
  const companion = fromBrowser ? browserCapture.status.manifest?.companion_capture : null;
  if (companion?.session_id) xhr.setRequestHeader("X-Companion-Session", companion.session_id);
  if (companion?.phone_capture_id) xhr.setRequestHeader("X-Phone-Capture-ID", companion.phone_capture_id);
  xhr.upload.addEventListener("progress", (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    uploadPercent.textContent = `${percent}%`;
    uploadBar.style.width = `${percent}%`;
  });
  xhr.addEventListener("load", async () => {
    uploadInProgress = false;
    if (!fromBrowser) sensorBundleInput.value = "";
    if (xhr.status < 200 || xhr.status >= 300) {
      let detail = `${xhr.status} sensor bundle upload failed`;
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (_) { /* keep HTTP detail */ }
      if (fromBrowser && browserCapture.status.bundle?.blob === file) {
        browserBundleUploadFailed = true;
        renderBrowserCaptureState(browserCapture.status);
      }
      uploadLabel.textContent = detail;
      toastMessage(detail);
      return;
    }
    let scan = JSON.parse(xhr.responseText);
    if (fromBrowser && browserCapture.status.bundle?.blob !== file) return;
    const intentResponse = await persistWalkIntent(scan.id, fromBrowser ? (activeBrowserIntent || currentWalkIntent()) : currentWalkIntent());
    if (intentResponse?.id === scan.id) scan = intentResponse;
    if (fromBrowser) activeBrowserIntent = null;
    if (fromBrowser) browserCapture.markBundleUploaded();
    rememberSelectedScan(scan.id);
    uploadPercent.textContent = "100%";
    uploadBar.style.width = "100%";
    uploadLabel.textContent = "Sensor bundle saved · preparing timestamped frames";
    toastMessage("Sensor capture auto-saved; frame preparation started");
    await refreshScans({ force: true });
    calibrationUI?.setScans(scans, { preferredId: scan.id });
    activateWorkspaceTab("library");
    setTimeout(() => uploadPanel.classList.add("hidden"), 1800);
  });
  xhr.addEventListener("error", () => {
    uploadInProgress = false;
    if (fromBrowser && browserCapture.status.bundle === browserBundle) {
      browserBundleUploadFailed = true;
      renderBrowserCaptureState(browserCapture.status);
    }
    uploadLabel.textContent = "Sensor bundle connection failed";
    toastMessage("Sensor bundle upload failed. Confirm the phone is still on home Wi-Fi.");
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

recordPhoneWalkButton.addEventListener("click", async () => {
  if (!captureIntentReady({ browser: !nativeCaptureUI.available })) return;
  if (nativeCaptureUI.available) {
    const intent = currentWalkIntent();
    nativeCaptureUI.capture(intent.mode, undefined, {
      target_scan_id: intent.target_scan_id,
      carry_protocol: intent.carry_protocol,
      accuracy_target_m: intent.accuracy_target_m,
      ...(intent.target_reference ? { target_reference: intent.target_reference } : {}),
    });
    return;
  }
  if (uploadInProgress) {
    toastMessage("Finish the current upload before starting another recording.");
    return;
  }
  if (browserCaptureBlocked("start a new browser recording")) return;
  browserCapturePanel.classList.remove("hidden");
  browserCapturePanel.scrollIntoView({ behavior: "smooth", block: "center" });
  if (["idle", "closed", "complete", "error"].includes(browserCapture.status.state)) {
    if (["failed", "stopped", "error"].includes(companionCapture.status.state) && !companionCapture.needsServerStop) companionCapture.reset();
    try {
      await browserCapture.open(browserPreview);
    } catch (error) {
      browserCapture.error = error;
      browserCapture.state = "error";
      renderBrowserCaptureState(browserCapture.status);
      return;
    }
    await companionCapture.listCameras().catch(() => {
      renderCompanionCaptureState(companionCapture.status);
      renderBrowserCaptureState(browserCapture.status);
    });
  }
});
companionCameraSelect?.addEventListener("change", (event) => {
  try {
    companionCapture.selectCamera(event.currentTarget.value);
  } catch (error) {
    event.currentTarget.value = "";
    toastMessage(error.message || String(error));
    renderCompanionCaptureState(companionCapture.status);
  }
});
browserStartButton.addEventListener("click", async () => {
  browserStartButton.disabled = true;
  let startedStatic = false;
  try {
    activeBrowserIntent = currentWalkIntent();
    const paired = captureMode === "path_refinement" || companionCapture.readyToStart;
    if (captureMode === "path_refinement" && !companionCapture.readyToStart) throw new Error("Choose an available static room camera before starting the paired recording.");
    if (paired) {
      const phoneCaptureId = companionCapture.reservePhoneCaptureId();
      await companionCapture.start({
        cameraId: companionCameraSelect.value,
        phoneCaptureId,
        browserCapture,
      });
      startedStatic = true;
      browserCapture.start({ captureId: phoneCaptureId, companionCapture: companionCapture.companionContext() });
      toastMessage("Static room video and phone camera are recording together");
    } else {
      browserCapture.start({ captureId: null, companionCapture: null });
      toastMessage("Phone reconstruction recording started; static pairing is optional");
    }
  } catch (error) {
    activeBrowserIntent = null;
    if (startedStatic || companionCapture.hasLiveSession || companionCapture.needsServerStop) {
      await companionCapture.stop("phone_start_failed").catch(() => {});
    }
    browserCapture.error = error;
    renderBrowserCaptureState(browserCapture.status);
  }
});
browserStopButton.addEventListener("click", async () => {
  browserStopButton.disabled = true;
  try {
    await browserCapture.stop("user");
  } catch (error) {
    browserCapture.error = error;
    renderBrowserCaptureState(browserCapture.status);
  }
});
browserCloseButton.addEventListener("click", async () => {
  if (browserCapture.isRecording || browserCapture.status.state === "stopping" || companionCapture.needsServerStop) {
    toastMessage("Stop and save the recording before closing this capture.");
    return;
  }
  await browserCapture.close();
  if (["failed", "stopped", "error"].includes(companionCapture.status.state) && !companionCapture.needsServerStop) companionCapture.reset();
  browserCapturePanel.classList.add("hidden");
});
browserMarkerButton?.addEventListener("click", async () => {
  try {
    await companionCapture.addMarker("user_event");
    toastMessage("Event marker saved with the paired capture");
  } catch (error) {
    toastMessage(`Marker retained for retry: ${error.message || error}`);
  }
});
browserMarkerRetryButton?.addEventListener("click", async () => {
  browserMarkerRetryButton.disabled = true;
  await companionCapture.retryMarkers();
  renderCompanionCaptureState(companionCapture.status);
  toastMessage(companionCapture.status.pendingMarkerCount ? "Some markers remain pending" : "Pending markers delivered");
});
companionStopRetryButton?.addEventListener("click", async () => {
  companionStopRetryButton.disabled = true;
  const result = await companionCapture.stop("retry");
  renderCompanionCaptureState(companionCapture.status);
  toastMessage(result.needsServerStop ? "Static capture still needs finalization; keep this page open." : "Static capture finalization acknowledged");
});
browserDownloadButton.addEventListener("click", () => {
  if (!browserCapture.downloadBundle()) toastMessage("The capture download is not available in this browser.");
});
browserRetryButton.addEventListener("click", () => {
  const bundle = browserCapture.status.bundle;
  if (bundle) uploadSensorBundle(bundle.blob, { fileName: bundle.fileName, fromBrowser: true });
});
browserDiscardButton.addEventListener("click", () => {
  if (uploadInProgress) {
    toastMessage("Wait for the current upload to finish before discarding this capture.");
    return;
  }
  browserCapture.discardBundle();
  activeBrowserIntent = null;
  if (["failed", "stopped", "error"].includes(companionCapture.status.state) && !companionCapture.needsServerStop) companionCapture.reset();
  browserBundleUploadFailed = false;
  browserCapturePanel.classList.add("hidden");
  toastMessage("The local browser capture was discarded.");
});
window.addEventListener("beforeunload", (event) => {
  if (!browserCapture.hasUnsavedWork() && !companionCapture.hasUnsavedWork()) return;
  event.preventDefault();
  event.returnValue = "";
});
existingInput.addEventListener("change", () => uploadVideo(existingInput.files?.[0]));
sensorBundleInput.addEventListener("change", () => uploadSensorBundle(sensorBundleInput.files?.[0]));
refreshButton.addEventListener("click", () => refreshScans({ force: true }));
newWalkButton.addEventListener("click", startNewWalk);
captureModeReconstruction?.addEventListener("change", () => setCaptureIntentMode("reconstruction"));
captureModePath?.addEventListener("change", () => setCaptureIntentMode("path_refinement"));
reconstructionTargetScan?.addEventListener("change", (event) => {
  reconstructionTargetId = event.currentTarget.value || "";
  if (reconstructionTargetId) localStorage.setItem(RECONSTRUCTION_TARGET_KEY, reconstructionTargetId);
  else localStorage.removeItem(RECONSTRUCTION_TARGET_KEY);
  syncCaptureIntentUi();
});
pathTargetScan?.addEventListener("change", (event) => {
  pathTargetId = event.currentTarget.value || "";
  if (pathTargetId) localStorage.setItem(PATH_TARGET_KEY, pathTargetId);
  else localStorage.removeItem(PATH_TARGET_KEY);
  syncCaptureIntentUi();
});
pathCameraSelect?.addEventListener("change", (event) => {
  pathCameraId = event.currentTarget.value || "";
  if (pathCameraId) localStorage.setItem(PATH_CAMERA_KEY, pathCameraId);
  else localStorage.removeItem(PATH_CAMERA_KEY);
  if (pathCameraId && nativeCaptureUI.available) {
    const snapshot = nativeCaptureUI.snapshot;
    const roomIndex = snapshot?.room_cameras?.findIndex((camera) => camera.camera_id === pathCameraId) ?? -1;
    if (roomIndex >= 0 && roomIndex !== snapshot.room_camera_index) {
      nativeCaptureUI.perform("configure", { room_camera_index: roomIndex });
    }
  }
  syncCaptureIntentUi();
});

if (newWalkMode) localStorage.removeItem("phoneScanSelected");
for (const button of document.querySelectorAll("[data-workspace-tab]")) {
  button.addEventListener("click", () => activateWorkspaceTab(button.dataset.workspaceTab));
  button.addEventListener("keydown", (event) => {
    const tabs = Array.from(document.querySelectorAll("[data-workspace-tab]"));
    const index = tabs.indexOf(button);
    const next = event.key === "ArrowRight" ? (index + 1) % tabs.length
      : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length
        : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : -1;
    if (next < 0) return;
    event.preventDefault();
    activateWorkspaceTab(tabs[next].dataset.workspaceTab);
    if (activeWorkspaceTab === tabs[next].dataset.workspaceTab) tabs[next].focus();
  });
}
if (isRoomWalkAndroid(window)) {
  document.querySelector("#browser-tools").open = false;
  recordPhoneWalkButton.textContent = "Open native camera";
}
syncCaptureModeUi();
syncCaptureIntentUi();
activateWorkspaceTab(!nativeCaptureUI.available && selectedId && !newWalkMode ? "library" : "capture");
nativeCaptureUI.refresh();
refreshHealth();
refreshScans({ force: true });
setInterval(refreshHealth, 30_000);
setInterval(refreshScans, 2_000);
