/* Native owns acquisition, paired-session leases, files and upload lifetime.
 * This module sends small commands and displays snapshots; it never transports
 * recordings, supplies acquisition timestamps, or starts a second static session.
 */
export const NATIVE_UI_SCHEMA = "roomwalk.native_state.v1";
export const NATIVE_ACTIONS = Object.freeze([
  "snapshot", "checkPhone", "checkConnection", "configure", "capture", "upload", "export", "share",
  "selectSaved", "repackage", "finalizePair", "uploadPhoneReport", "startImu",
  "stopImu", "cancelUpload", "retryUpload",
]);
export const WALK_INTENT_SCHEMA = "roomwalk.walk_intent.v1";
const PATH_REFERENCE_KEYS = Object.freeze(["kind", "prior_id", "manifest_sha256", "camera_id", "frame_binding_sha256"]);
const SAFE_PRIOR_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$/;
const SAFE_CAMERA_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$/;
const SAFE_SCAN_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const SHA256 = /^[0-9a-f]{64}$/;

function normalizeCaptureMode(mode) {
  return mode === "walk" ? "reconstruction" : mode;
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function validatePathReferenceSelection(value) {
  if (!isRecord(value)) throw new Error("The retained PCF selection is invalid.");
  const keys = Object.keys(value);
  if (keys.length !== PATH_REFERENCE_KEYS.length || keys.some((key) => !PATH_REFERENCE_KEYS.includes(key))) {
    throw new Error("The retained PCF selection contains unsupported fields.");
  }
  if (value.kind !== "scene_prior_pcf") throw new Error("The retained PCF selection kind is unsupported.");
  if (typeof value.prior_id !== "string" || !SAFE_PRIOR_ID.test(value.prior_id)) throw new Error("The retained PCF prior_id is invalid.");
  if (typeof value.camera_id !== "string" || !SAFE_CAMERA_ID.test(value.camera_id)) throw new Error("The retained PCF camera_id is invalid.");
  for (const key of ["manifest_sha256", "frame_binding_sha256"]) {
    if (typeof value[key] !== "string" || !SHA256.test(value[key])) {
      throw new Error(`The retained PCF ${key} must be a lowercase SHA-256 digest.`);
    }
  }
  return Object.fromEntries(PATH_REFERENCE_KEYS.map((key) => [key, value[key]]));
}

export function pathSelectionLabel(selection) {
  const value = validatePathReferenceSelection(selection);
  return `Noesis PCF · ${value.camera_id} · prior ${value.prior_id}`;
}

export function pathReferenceForScan(scan) {
  if (!isRecord(scan) || !Object.prototype.hasOwnProperty.call(scan, "path_reference")) {
    return { status: "base", selection: null, label: "Original reconstruction · no retained PCF reference" };
  }
  const reference = scan.path_reference;
  if (!isRecord(reference) || reference.status !== "available") {
    const reason = typeof reference?.reason === "string" && reference.reason.trim()
      ? reference.reason.trim().slice(0, 240)
      : "No retained PCF reference is available for this reconstruction.";
    return { status: "unavailable", selection: null, reason, label: `Noesis PCF unavailable · ${reason}` };
  }
  try {
    const selection = validatePathReferenceSelection(reference.selection);
    return {
      status: "available",
      selection,
      // Keep the exact prior identity visible even if the server's display label
      // is shortened to a camera name.
      label: pathSelectionLabel(selection),
      source_scan_id: typeof reference.source_scan_id === "string" ? reference.source_scan_id : "",
    };
  } catch (error) {
    const reason = `The retained PCF reference is invalid: ${error.message}`.slice(0, 240);
    return { status: "unavailable", selection: null, reason, label: `Noesis PCF unavailable · ${reason}` };
  }
}

export function normalizeWalkIntent(intent = {}) {
  if (!isRecord(intent)) throw new Error("Walk intent must be an object.");
  if (intent.schema !== undefined && intent.schema !== WALK_INTENT_SCHEMA) throw new Error("Walk intent schema is unsupported.");
  const mode = normalizeCaptureMode(intent.mode);
  if (!["reconstruction", "path_refinement"].includes(mode)) throw new Error("Choose reconstruction or path refinement.");
  const target = intent.target_scan_id === null || intent.target_scan_id === undefined || intent.target_scan_id === ""
    ? null : intent.target_scan_id;
  if (target !== null && (typeof target !== "string" || !SAFE_SCAN_ID.test(target))) throw new Error("The target reconstruction ID is invalid.");
  const normalized = {
    schema: WALK_INTENT_SCHEMA,
    mode,
    target_scan_id: target,
    carry_protocol: mode === "path_refinement" ? "close_body" : "coverage",
    accuracy_target_m: 0.1,
  };
  if (Object.prototype.hasOwnProperty.call(intent, "target_reference")) {
    if (mode !== "path_refinement") throw new Error("A retained PCF selection is valid only for path refinement.");
    normalized.target_reference = validatePathReferenceSelection(intent.target_reference);
  }
  return normalized;
}

export function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

function nonnegative(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") return null;
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export function readableBytes(value) {
  const bytes = nonnegative(value);
  if (bytes === null) return "Not reported";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = bytes ? Math.min(4, Math.floor(Math.log(bytes) / Math.log(1000))) : 0;
  return `${(bytes / 1000 ** Math.max(0, index)).toLocaleString("en-US", { maximumFractionDigits: index > 0 ? 1 : 0 })} ${units[Math.max(0, index)]}`;
}

export function readableDuration(value) {
  const seconds = nonnegative(value);
  if (seconds === null) return "Not reported";
  const rounded = Math.round(seconds), hours = Math.floor(rounded / 3600), minutes = Math.floor(rounded / 60) % 60;
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${rounded % 60}s` : `${rounded}s`;
}

export function captureGuidanceMarkup(snapshot = {}) {
  const sensors = snapshot.imu_telemetry;
  if (sensors && typeof sensors === "object" && (snapshot.imu_preparing || snapshot.imu_active || (!snapshot.active && !snapshot.capture_guidance))) {
    const elapsed = nonnegative(sensors.elapsed_seconds), target = nonnegative(sensors.expected_duration_s);
    const preparing = snapshot.imu_preparing === true;
    const title = preparing ? "Set the phone down" : snapshot.imu_active ? "Recording stationary sensors" : "Sensor recording finished";
    const counts = [["Accelerometer", sensors.accel_samples], ["Gyroscope", sensors.gyro_samples]];
    return `<strong>${title}</strong><div class="capture-duration">${preparing ? `Starting in ${nonnegative(sensors.countdown_seconds) ?? "…"}s` : `${elapsed === null ? "—" : Math.floor(elapsed)} / ${target ?? "—"}s`}</div>
      <p>${preparing ? "Not recording yet. Leave the phone untouched." : snapshot.imu_active ? "Keep still. Recording stops automatically; no scrolling needed." : "Ready to upload. Processing checks whether this take is usable."}</p>
      ${!preparing && elapsed !== null && target > 0 ? `<progress class="transfer-progress" max="${target}" value="${Math.min(elapsed, target)}" aria-label="Recorded sensor duration"></progress>` : ""}
      <dl class="capture-counts">${counts.map(([label, value]) => `<div><dt>${label}</dt><dd>${nonnegative(value) === null ? "Waiting…" : Number(value).toLocaleString("en-US")} samples</dd></div>`).join("")}</dl>`;
  }
  const guide = snapshot.capture_guidance;
  if (!guide || typeof guide !== "object" || !["walk", "reconstruction", "path_refinement", "camera", "imu", "noise"].includes(guide.mode)) {
    return escapeHtml(snapshot.imu_active ? snapshot.imu_progress || "Sensors-only recording in progress. Keep the phone untouched on a stable surface."
      : snapshot.active ? "Recording on the phone. Follow the native viewer; elapsed capture time is not reported here."
        : "Capture progress is reported by the native recorder, not estimated by this page.");
  }
  // Never count page time or convert duration/completed into a quality gate.
  const elapsed = nonnegative(guide.elapsed_seconds), target = nonnegative(guide.target_seconds);
  const percent = elapsed !== null && target > 0 ? Math.min(100, elapsed / target * 100) : null;
  const pathInstruction = guide.mode === "path_refinement"
    ? "Carry the phone against your own torso with elbows tucked and stable. Turn and move your whole body with the phone. Do not film another person, sweep the phone with your arm, vary phone-to-body distance, or walk a nearby body route."
    : null;
  const title = guide.mode === "path_refinement" ? "Self-carried path refinement" : String(guide.title || "Native capture guidance").slice(0, 200);
  const assurance = guide.mode === "path_refinement"
    ? "Duration is a capture target; it is not proof of 10 cm path accuracy."
    : guide.mode === "reconstruction"
      ? "Duration guides room coverage; it is not a reconstruction-quality result."
      : "Duration is a capture target, not evidence that calibration passed.";
  return `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(pathInstruction || String(guide.instruction || "Follow the native viewer.").slice(0, 1500))}</p>
    ${elapsed !== null ? `<p>${escapeHtml(readableDuration(elapsed))} elapsed${target > 0 ? ` · ${escapeHtml(readableDuration(target))} target` : ""}${guide.completed === true ? " · Capture finished; upload and verify the data" : ""}</p>` : ""}
    ${percent !== null ? `<progress class="transfer-progress" max="100" value="${percent}" aria-label="Capture duration, not a quality result"></progress>` : ""}<p>${assurance}</p>`;
}

export function transferPresentation(transfer = {}) {
  const state = transfer.state || "idle";
  const sent = nonnegative(transfer.sent_bytes), total = nonnegative(transfer.total_bytes);
  const speed = nonnegative(transfer.bytes_per_second), elapsed = nonnegative(transfer.elapsed_ms);
  const receipt = transfer.receipt && typeof transfer.receipt === "object" ? transfer.receipt : null;
  const titles = { idle: "Ready to upload", queued: "Upload queued", connecting: "Connecting to Noesis", finalizing: "Finalizing the paired recording", uploading: "Uploading recording", validating: "Sent · waiting for server receipt", cancelling: "Cancelling upload", cancelled: "Upload cancelled", interrupted: "Upload interrupted", failed: "Upload needs attention", complete: "Transfer finished · receipt not reported" };
  let title = titles[state] || "Transfer status", tone = ["failed", "interrupted"].includes(state) ? "error" : "neutral";
  let description = state === "idle" ? "Upload the selected ZIP to process it on Noesis. The original stays on this phone."
    : state === "validating" ? "Bytes have been sent. Wait for the storage receipt; server acceptance has not been confirmed yet."
      : "The original recording stays on this phone. Transfer and server validation are separate steps.";
  if (state === "complete" && receipt) {
    tone = "success";
    if (transfer.imu === true) {
      title = "Stationary recording stored";
      description = "Storage accepted. Choose this recording in Setup → stationary diagnostics to process it. Noise qualification has not been established by this upload.";
    } else if (receipt.validation_status === "pending" || receipt.status === "importing_capture") {
      title = "Storage accepted · import pending";
      tone = "warning";
      description = "Noesis has stored the archive. Import and evidence validation are still pending; storage is not calibration acceptance.";
    } else if (receipt.status === "import_failed") {
      title = "Storage accepted · import needs attention";
      tone = "warning";
      description = "The archive is retained, but server import failed. Open the server capture for its validation errors.";
    } else {
      title = "Storage accepted";
      description = "The server receipt confirms storage. Review the server capture for current import and calibration qualification; the phone copy remains retained.";
    }
  }
  const percent = sent !== null && total !== null && total > 0 ? Math.min(100, Math.max(0, sent / total * 100)) : null;
  const stats = [];
  if (sent !== null || total !== null) stats.push(["Transferred", `${readableBytes(sent)}${total !== null ? ` / ${readableBytes(total)}` : ""}`]);
  if (speed !== null && speed > 0) stats.push([transfer.active ? "Average speed" : "Average transfer speed", `${readableBytes(speed)}/s`]);
  if (elapsed !== null) stats.push(["Elapsed", readableDuration(elapsed / 1000)]);
  if (state === "uploading" && speed > 0 && sent !== null && total > sent) stats.push(["Estimated remaining", readableDuration((total - sent) / speed)]);
  return { state, title, tone, description, percent, stats, archive: typeof transfer.archive_name === "string" ? transfer.archive_name : "", error: typeof transfer.error === "string" ? transfer.error.slice(0, 2000) : "", active: transfer.active === true };
}

export function transferCardMarkup(transfer) {
  const view = transferPresentation(transfer);
  return `<div class="result-heading"><div><span class="eyebrow">Latest phone transfer</span><h3>${escapeHtml(view.title)}</h3></div><span class="result-badge ${view.tone}">${escapeHtml(view.state === "complete" ? "Transfer finished" : view.state)}</span></div>
    ${view.archive ? `<p class="transfer-filename">${escapeHtml(view.archive)}</p>` : ""}
    ${view.percent !== null ? `<div class="progress-header"><span>${view.active ? "Bytes transferred" : "Last transfer progress"}</span><b>${Math.floor(view.percent)}%</b></div><progress class="transfer-progress" max="100" value="${view.percent}" aria-label="Bytes transferred"></progress>` : view.active ? '<progress class="transfer-progress" aria-label="Transfer is working"></progress>' : ""}
    ${view.stats.length ? `<dl class="result-metrics">${view.stats.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
    <p class="tool-note">${escapeHtml(view.description)}</p>${view.error ? `<p class="result-error" role="status">${escapeHtml(view.error)}</p>` : ""}`;
}

export function isRoomWalkAndroid(platform = globalThis) {
  return platform.__ROOMWALK_ANDROID__ === true
    || /(?:^|\s)RoomWalkAndroid\/[\w.-]+(?:\s|$)/.test(platform.navigator?.userAgent || "");
}

export function nativeCommandUrl(action, args = {}) {
  if (!NATIVE_ACTIONS.includes(action)) throw new Error("Unknown native action.");
  if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Native action arguments must be an object.");
  const url = `roomwalk-native://action?request=${encodeURIComponent(JSON.stringify({ action, args }))}`;
  if (url.length > 32_768) throw new Error("Native request is too large. Recordings must remain in native storage.");
  return url;
}

export function parseNativeSnapshot(value) {
  const snapshot = typeof value === "string" ? JSON.parse(value) : value;
  if (!snapshot || typeof snapshot !== "object" || Array.isArray(snapshot) || snapshot.schema !== NATIVE_UI_SCHEMA) {
    throw new Error("RoomWalk received an unsupported native status message.");
  }
  return {
    ...snapshot,
    cameras: Array.isArray(snapshot.cameras) ? snapshot.cameras : [],
    room_cameras: Array.isArray(snapshot.room_cameras) ? snapshot.room_cameras : [],
    saved: Array.isArray(snapshot.saved) ? snapshot.saved : [],
    enabled: snapshot.enabled && typeof snapshot.enabled === "object" && !Array.isArray(snapshot.enabled) ? snapshot.enabled : {},
  };
}

export function validateNativeConfiguration(values) {
  let server;
  try { server = new URL(String(values.server || "").trim()); } catch (_) { throw new Error("Enter the trusted HTTPS server address."); }
  if (server.protocol !== "https:" || server.username || server.password || server.search || server.hash || server.pathname !== "/") {
    throw new Error("Use an HTTPS server origin, without credentials, a path, query, or fragment.");
  }
  const configuration = { server: server.origin };
  for (const key of ["camera_index", "room_camera_index"]) {
    const value = Number(values[key]);
    if (!Number.isInteger(value) || value < -1 || value > 255) throw new Error("Select a valid camera from the phone's current inventory.");
    configuration[key] = value;
  }
  const minutes = Number(values.imu_minutes);
  if (!Number.isInteger(minutes) || minutes < 1 || minutes > 5) throw new Error("The sensor diagnostic must be between 1 and 5 minutes.");
  configuration.imu_minutes = minutes;
  return configuration;
}

function describe(value, limit = 16_384) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return text.length > limit ? `${text.slice(0, limit)}\n… See the retained native evidence for the full record.` : text;
}

function cameraOptions(cameras, selected, { room = false } = {}) {
  return `<option value="-1" ${selected === -1 ? "selected" : ""}>${room ? "None · phone-only reconstruction" : "Choose camera…"}</option>` + cameras.slice(0, 256).map((camera, index) => {
    const label = typeof camera === "string" ? camera : camera?.label || camera?.name || camera?.camera_id || camera?.id || `Camera ${index + 1}`;
    const capability = room ? "" : camera?.full_walk_available === true ? " · full walk available" : camera?.test_available === true ? " · test available" : " · capability not ready";
    return `<option value="${index}" ${index === selected ? "selected" : ""}>${escapeHtml(label)}${capability}</option>`;
  }).join("");
}

export function savedCaptureCards(snapshot) {
  const selected = typeof snapshot.selected_capture === "string" ? snapshot.selected_capture : snapshot.selected_capture?.id;
  return snapshot.saved.filter((item) => item && typeof item.id === "string").slice(0, 256).map((item) => {
    const intent = item.walk_intent || item.capture?.walk_intent;
    const intentLabel = intent?.mode === "path_refinement" ? "Path refinement" : intent?.mode === "reconstruction" ? `Reconstruction${intent.target_scan_id ? " · add views" : ""}` : null;
    const label = intentLabel || (item.calibration_mode === "camera" ? item.short_test ? "Camera timing test · not calibration" : "Camera calibration" : item.calibration_mode === "imu" ? item.short_test ? "Motion timing test · not calibration" : "Motion calibration" : item.kind === "imu_diagnostic" ? "Stationary sensors" : item.kind === "walk" ? "Room walk" : item.kind === "calibration" ? "Calibration · settings need review" : "Phone report / recording");
    const target = typeof intent?.target_scan_id === "string" && intent.target_scan_id ? ` · target ${intent.target_scan_id}` : "";
    let reference = "";
    if (intent?.mode === "path_refinement" && intent.target_reference) {
      try { reference = ` · ${pathSelectionLabel(intent.target_reference)}`; } catch (_) { reference = " · retained PCF selection needs review"; }
    }
    return `<button type="button" data-native-saved="${escapeHtml(item.id)}" aria-pressed="${selected === item.id}" ${snapshot.enabled.selectSaved === true ? "" : "disabled"}><strong>${escapeHtml(item.name || item.id)}</strong><small>${escapeHtml(label)}${escapeHtml(target)}${escapeHtml(reference)}${item.upload_ready ? " · Ready to upload" : ""}</small></button>`;
  }).join("") || '<p class="tool-note">No retained captures reported by this phone yet.</p>';
}

export class NativeBridge {
  constructor({ platform = globalThis, onSnapshot = () => {}, onError = () => {}, navigate, now = () => Date.now() } = {}) {
    this.platform = platform;
    this.onSnapshot = onSnapshot;
    this.onError = onError;
    this.now = now;
    this.pendingAction = null;
    this.sentAt = 0;
    this.navigate = navigate || ((url) => { this.platform.location.href = url; });
    this.receive = (payload) => {
      if (!this.available) return false;
      try { const state = parseNativeSnapshot(payload); this.pendingAction = null; this.onSnapshot(state); return true; }
      catch (error) { this.onError(error); return false; }
    };
    // Install before requesting the initial snapshot. The user-agent token is
    // available even if onPageStarted's __ROOMWALK_ANDROID__ injection races us.
    this.platform.RoomWalkNative = { receive: this.receive };
  }

  get available() { return isRoomWalkAndroid(this.platform); }

  send(action, args = {}) {
    if (!this.available) throw new Error("This action requires the RoomWalk Android app.");
    const url = nativeCommandUrl(action, args);
    // No action queue or automatic retry: two main-frame navigations in one
    // stack can replace each other. Await a native snapshot before another action.
    // A manual snapshot after two seconds is read-only recovery, never a resend.
    if (this.pendingAction) {
      if (action === "snapshot" && this.now() - this.sentAt < 2000) return false;
      if (action !== "snapshot") throw new Error("Waiting for the phone to confirm the previous action. Use Refresh phone if its status does not update.");
    }
    this.pendingAction = action;
    this.sentAt = this.now();
    try { this.navigate(url); } catch (error) { this.pendingAction = null; throw error; }
    return true;
  }
}

export class NativeCaptureUI {
  constructor({ captureRoot, libraryRoot, platform = globalThis, onState = () => {}, onUploaded = () => {}, onPackaged = () => {}, onOpenLibrary = () => {}, onError = () => {}, canCapture = () => true } = {}) {
    this.captureRoot = captureRoot;
    this.libraryRoot = libraryRoot;
    this.platform = platform;
    this.onState = onState;
    this.onUploaded = onUploaded;
    this.onPackaged = onPackaged;
    this.onOpenLibrary = onOpenLibrary;
    this.awaitingPackaged = false;
    this.captureWorkObserved = false;
    this.awaitingCaptureId = "";
    this.captureStartArtifact = "";
    this.onError = onError;
    this.canCapture = canCapture;
    this.snapshot = null;
    this.configDirty = false;
    this.pendingConfiguration = null;
    this.intentError = null;
    this.captureIntent = { schema: WALK_INTENT_SCHEMA, mode: "reconstruction", target_scan_id: null, carry_protocol: "coverage", accuracy_target_m: 0.1 };
    this.bridge = new NativeBridge({ platform, onSnapshot: (snapshot) => this.receive(snapshot), onError });
    this.renderShell();
    this.bind();
    this._onVisibility = () => {
      if (this.platform.document?.visibilityState === "visible") this.refresh();
    };
    this.platform.document?.addEventListener("visibilitychange", this._onVisibility);
    this.updateVisibility();
  }

  get available() { return this.bridge.available; }
  enabled(action) { return action === "snapshot" ? this.available : this.available && !this.bridge.pendingAction && this.snapshot?.enabled?.[action] === true; }
  hasActiveWork() { return Boolean(this.snapshot?.active || this.snapshot?.busy || this.snapshot?.imu_active || this.snapshot?.transfer?.active); }

  setCaptureIntent(intent = {}) {
    const normalized = normalizeWalkIntent(intent);
    this.intentError = null;
    this.captureIntent = normalized;
    const mode = normalized.mode;
    const summary = this.captureRoot?.querySelector("#native-intent-summary");
    if (summary) summary.textContent = mode === "path_refinement"
      ? `Path refinement · ${normalized.target_reference ? pathSelectionLabel(normalized.target_reference) : "Original reconstruction · no retained PCF reference"} · phone against own torso · elbows tucked · target ${normalized.target_scan_id || "not selected"} · paired camera required`
      : `Reconstruction · coverage${normalized.target_scan_id ? ` · add views to ${normalized.target_scan_id}` : " · new room"}`;
    this.updateButtons();
  }

  updateVisibility() {
    this.captureRoot?.classList.toggle("hidden", !this.available);
    this.libraryRoot?.classList.toggle("hidden", !this.available);
    this.platform.document?.body?.classList.toggle("roomwalk-android", this.available);
  }

  refresh() {
    this.updateVisibility();
    if (this.available) this.perform("snapshot");
  }

  perform(action, args = {}) {
    try {
      if (!this.enabled(action)) throw new Error("That action is not ready. Check the phone status and refresh if needed.");
      const sent = this.bridge.send(action, args);
      if (sent && ["capture", "startImu"].includes(action)) {
        this.awaitingPackaged = true;
        this.captureWorkObserved = false;
        this.awaitingCaptureId = "";
        this.captureStartArtifact = this.artifactKey(this.snapshot);
      }
      if (sent && action === "selectSaved") this.awaitingPackaged = false;
      this.updateButtons();
      if (this.snapshot) this.onState(this.snapshot);
      return sent;
    } catch (error) { this.onError(error); return false; }
  }

  capture(mode = "reconstruction", board, { board_geometry_confirmed = false, camera_calibration_id, noise_calibration_id, target_scan_id, carry_protocol, accuracy_target_m, target_reference } = {}) {
    try {
      mode = normalizeCaptureMode(mode);
      if (!["reconstruction", "path_refinement", "camera", "imu"].includes(mode)) throw new Error("Choose a supported capture mode.");
      if (mode === "path_refinement" && !(typeof target_scan_id === "string" && target_scan_id.trim())) throw new Error("Choose the existing reconstruction before starting path refinement.");
      if (this.configDirty) throw new Error("Apply the edited phone setup before opening the camera.");
      if (!this.canCapture(mode)) return false;
      if (target_reference !== undefined && this.snapshot?.supports_target_reference !== true) {
        throw new Error("This RoomWalk app cannot preserve the selected PCF reference. Update the companion before recording this path.");
      }
      if (["reconstruction", "path_refinement"].includes(mode)) this.setCaptureIntent({ mode, target_scan_id, carry_protocol, accuracy_target_m, ...(target_reference === undefined ? {} : { target_reference }) });
      const focus = mode === "imu" ? this.snapshot?.cameras?.[this.snapshot.camera_index]?.focus_control : null;
      return this.perform("capture", { mode, ...(target_scan_id ? { target_scan_id } : {}), ...(carry_protocol ? { carry_protocol } : {}), ...(Number.isFinite(Number(accuracy_target_m)) ? { accuracy_target_m: 0.1 } : {}), ...(target_reference === undefined ? {} : { target_reference }), ...(board ? { board, board_geometry_confirmed: board_geometry_confirmed === true } : {}), ...(camera_calibration_id ? { camera_calibration_id } : {}), ...(noise_calibration_id ? { noise_calibration_id } : {}), ...(focus ? { calibration_focus_guard: focus } : {}) });
    } catch (error) { this.onError(error); return false; }
  }

  startImu(minutes, { stationaryNoise = false } = {}) {
    const maximum = stationaryNoise ? 180 : 5;
    if (!Number.isInteger(minutes) || minutes < 1 || minutes > maximum) {
      this.onError(new Error(`Choose 1–${maximum} minutes for this sensor recording protocol.`));
      return false;
    }
    if (!this.canCapture("noise")) return false;
    return this.perform("startImu", { minutes });
  }

  renderShell() {
    if (this.captureRoot) this.captureRoot.innerHTML = `
      <div class="native-capture-heading"><span class="eyebrow">Native camera + IMU</span><h2>RoomWalk recorder</h2><p id="native-intent-summary" class="native-intent-summary">Reconstruction · coverage · new room</p></div>
      <div class="tool-actions native-primary-actions"><button id="native-check-primary" type="button" class="primary-button" data-native-action="checkPhone" disabled>Check phone</button><button id="native-capture-primary" type="button" class="primary-button hidden" data-native-action="capture" disabled>Capture room walk</button></div>
      <div class="native-brief"><strong id="native-status" role="status">Connecting to the native recorder…</strong><p id="native-camera-summary">Check your phone to find its recording camera.</p><button id="native-open-library" type="button" class="secondary-button hidden">Open saved capture in Library</button></div>
      <details id="native-setup" class="tool-details"><summary>Connection and camera selection</summary>
      <form id="native-configuration">
        <div class="tool-grid">
          <label class="tool-field tool-wide"><span>Trusted Noesis server</span><input id="native-server" class="text-input" type="url" inputmode="url" autocomplete="off" spellcheck="false" required placeholder="https://your-noesis-host:8789" /></label>
          <label class="tool-field"><span>Phone camera</span><select id="native-camera" class="text-input"><option value="-1">Check phone to load cameras…</option></select></label>
          <label class="tool-field"><span>Static room camera</span><select id="native-room-camera" class="text-input"><option value="-1">Connect to load room cameras…</option></select></label>
        </div>
        <p id="native-room-status" class="tool-note">No room-camera status yet.</p>
        <div class="tool-actions"><button type="submit" class="secondary-button" data-native-action="configure" disabled>Apply setup</button><button type="button" class="secondary-button" data-native-action="checkConnection" disabled>Check connection</button></div>
      </form></details>
      <p id="native-setup-note" class="tool-note">Check the phone, then open the camera. Saved recordings appear in Library.</p>
      <details id="native-status-details" class="tool-details"><summary>Phone status and refresh</summary><p id="native-detail" class="tool-note">Native capture remains available without the server. Paired room walks need the selected room camera.</p><button type="button" class="ghost-button" data-native-action="snapshot">Refresh phone</button></details>
        <details class="tool-details"><summary>Advanced phone diagnostics</summary>
          <p class="tool-note">This short sensor-only recording checks sensor behavior. It is not camera–IMU calibration or a prerequisite for a walk.</p>
          <label class="tool-field"><span>Sensor diagnostic duration (minutes)</span><input id="native-imu-minutes" class="text-input" type="number" min="1" max="5" step="1" value="1" /></label>
          <div class="tool-actions"><button type="button" class="secondary-button" data-native-action="startImu" disabled>Record sensors only</button><button type="button" class="danger-button" data-native-action="stopImu" disabled>Stop sensor check</button><button type="button" class="ghost-button" data-native-action="uploadPhoneReport" disabled>Upload phone report</button></div>
          <p id="native-imu-progress" class="tool-note" role="status">No sensor diagnostic running.</p>
        </details>`;
    if (this.libraryRoot) this.libraryRoot.innerHTML = `
      <div class="tool-heading"><div><span class="eyebrow">Saved on this phone</span><h2>Retained captures</h2><p>Upload a retained recording to process it on Noesis. Your original stays on the phone.</p></div><button type="button" class="ghost-button" data-native-action="snapshot">Refresh phone</button></div>
      <section id="native-selected-card" class="tool-status selected-capture-card"><span class="eyebrow">Selected recording</span><h3 id="native-artifact">No capture selected</h3><p id="native-library-status">Native files and server-side walks are separate retained copies.</p>
        <div class="tool-actions"><button type="button" class="primary-button" data-native-action="upload" disabled>Upload selected capture</button><button type="button" class="secondary-button" data-native-action="export" disabled>Export</button></div>
        <details class="tool-details"><summary>Share and recovery options</summary><div class="tool-actions"><button type="button" class="secondary-button" data-native-action="share" disabled>Share</button><button type="button" class="ghost-button" data-native-action="repackage" disabled>Package again</button><button type="button" class="ghost-button" data-native-action="finalizePair" disabled>Retry paired finalization</button></div></details>
      </section>
      <section class="native-transfer-card" aria-label="Phone upload progress"><div id="native-transfer-summary"></div><div class="tool-actions"><button type="button" class="secondary-button hidden" data-native-action="retryUpload" disabled>Retry this transfer</button><button type="button" class="danger-button hidden" data-native-action="cancelUpload" disabled>Cancel upload</button></div></section>
      <details class="tool-details" id="native-saved-picker"><summary>Choose another saved capture</summary><div id="native-saved-list" class="native-saved-list"></div></details>
      <details class="tool-details"><summary>Advanced · raw transfer receipt</summary><pre id="native-transfer" class="tool-report">No transfer reported.</pre></details>`;
    const server = this.captureRoot?.querySelector("#native-server");
    if (server && this.platform.location?.protocol === "https:") server.value = this.platform.location.origin;
  }

  configuration() {
    const value = (selector) => this.captureRoot.querySelector(selector).value;
    return validateNativeConfiguration({ server: value("#native-server"), camera_index: value("#native-camera"), room_camera_index: value("#native-room-camera"), imu_minutes: value("#native-imu-minutes") });
  }

  bind() {
    this.captureRoot?.querySelector("#native-open-library")?.addEventListener("click", () => this.onOpenLibrary());
    const form = this.captureRoot?.querySelector("#native-configuration");
    form?.addEventListener("input", () => { this.configDirty = true; this.updateButtons(); });
    form?.addEventListener("change", () => { this.configDirty = true; this.updateButtons(); });
    form?.addEventListener("submit", (event) => {
      event.preventDefault();
      try {
        const configuration = this.configuration();
        this.pendingConfiguration = configuration;
        if (!this.perform("configure", configuration)) this.pendingConfiguration = null;
      } catch (error) { this.onError(error); }
    });
    for (const root of [this.captureRoot, this.libraryRoot]) root?.addEventListener("click", (event) => {
      const saved = event.target.closest?.("[data-native-saved]");
      if (saved && root.contains(saved) && !saved.disabled) { this.perform("selectSaved", { id: saved.dataset.nativeSaved }); return; }
      const button = event.target.closest?.("[data-native-action]");
      if (!button || !root.contains(button) || button.disabled) return;
      const action = button.dataset.nativeAction;
      if (action === "configure") return; // Form validation and submit own this action.
      if (action === "capture") { if (!this.captureIntent) throw new Error("The retained walk intent needs review before capture."); this.capture(this.captureIntent.mode, undefined, this.captureIntent); return; }
      if (action === "startImu") {
        const minutes = Number(this.captureRoot.querySelector("#native-imu-minutes").value);
        this.startImu(minutes);
        return;
      }
      this.perform(action);
    });
  }

  receive(snapshot) {
    const previous = this.snapshot;
    this.bridge.pendingAction = null;
    this.snapshot = snapshot;
    if (snapshot.walk_intent && typeof snapshot.walk_intent === "object") {
      try { this.setCaptureIntent(snapshot.walk_intent); }
      catch (error) { this.intentError = error; this.captureIntent = null; this.onError(error); }
    }
    this.updateVisibility();
    if (this.pendingConfiguration) {
      const requested = this.pendingConfiguration;
      // Older hosts may not echo imu_minutes; camera and server changes still
      // require an exact acknowledgement before replacing an edited form.
      if (snapshot.server === requested.server && snapshot.camera_index === requested.camera_index && snapshot.room_camera_index === requested.room_camera_index
          && (snapshot.imu_minutes === undefined || Number(snapshot.imu_minutes) === requested.imu_minutes)) {
        this.configDirty = false;
        this.pendingConfiguration = null;
      }
    }
    const setText = (root, selector, value) => { const element = root?.querySelector(selector); if (element) element.textContent = describe(value); };
    setText(this.captureRoot, "#native-status", snapshot.status || "Native status received");
    setText(this.captureRoot, "#native-detail", snapshot.detail);
    setText(this.captureRoot, "#native-room-status", snapshot.room_camera_status || "No room-camera status reported.");
    setText(this.captureRoot, "#native-imu-progress", snapshot.imu_active ? snapshot.imu_progress || "Recording native sensor evidence…" : snapshot.imu_progress || "No sensor diagnostic running.");
    const label = (camera) => typeof camera === "string" ? camera : camera?.label || camera?.name || camera?.camera_id || camera?.id || "Not selected";
    setText(this.captureRoot, "#native-camera-summary", snapshot.cameras.length ? `Phone: ${label(snapshot.cameras[snapshot.camera_index])} · Room: ${label(snapshot.room_cameras[snapshot.room_camera_index])}` : "Start with Check phone. Connection settings are below when you need them.");
    if (snapshot.cameras.length && (!previous?.cameras?.length || this.configDirty) && snapshot.camera_index < 0) {
      const setup = this.captureRoot?.querySelector("#native-setup");
      if (setup) setup.open = true;
    }
    if (snapshot.status !== previous?.status && /needs attention|could not|failed|error/i.test(snapshot.status || "")) {
      const details = this.captureRoot?.querySelector("#native-status-details");
      if (details) details.open = true;
    }
    if (!this.configDirty && this.captureRoot) {
      if (typeof snapshot.server === "string") this.captureRoot.querySelector("#native-server").value = snapshot.server;
      this.captureRoot.querySelector("#native-camera").innerHTML = cameraOptions(snapshot.cameras, snapshot.camera_index);
      this.captureRoot.querySelector("#native-room-camera").innerHTML = cameraOptions(snapshot.room_cameras, snapshot.room_camera_index, { room: true });
      // Retained historical sessions must not widen the short default.
      const diagnosticMinutes = Number(snapshot.imu_minutes);
      if (Number.isInteger(diagnosticMinutes) && diagnosticMinutes >= 1 && diagnosticMinutes <= 5) this.captureRoot.querySelector("#native-imu-minutes").value = diagnosticMinutes;
    }
    const savedList = this.libraryRoot?.querySelector("#native-saved-list");
    if (savedList) savedList.innerHTML = savedCaptureCards(snapshot);
    const artifact = snapshot.artifact;
    setText(this.libraryRoot, "#native-artifact", artifact?.name ? `${artifact.name} · ${readableBytes(artifact.bytes)}` : snapshot.selected_capture ? "Capture selected · no packaged archive reported" : "No capture selected");
    setText(this.libraryRoot, "#native-library-status", snapshot.detail || snapshot.status);
    setText(this.libraryRoot, "#native-transfer", snapshot.transfer || "No transfer reported.");
    const summary = this.libraryRoot?.querySelector("#native-transfer-summary");
    if (summary) summary.innerHTML = transferCardMarkup(snapshot.transfer || {});
    summary?.closest(".native-transfer-card")?.classList.toggle("hidden", !snapshot.transfer?.state || snapshot.transfer.state === "idle");
    this.captureRoot?.querySelector("#native-open-library")?.classList.toggle("hidden", !snapshot.selected_capture || this.hasActiveWork());
    if (!snapshot.selected_capture && snapshot.saved.length) {
      const picker = this.libraryRoot?.querySelector("#native-saved-picker");
      if (picker) picker.open = true;
    }
    this.updateButtons();
    this.onState(snapshot);
    const scanId = snapshot.last_scan_id;
    if (previous && typeof scanId === "string" && scanId && scanId !== previous.last_scan_id) this.onUploaded(scanId);
    if (snapshot.active || snapshot.imu_active) {
      if (!this.awaitingPackaged) this.captureStartArtifact = this.artifactKey(previous);
      this.awaitingPackaged = true;
      this.awaitingCaptureId = snapshot.selected_capture;
      this.captureWorkObserved = true;
    }
    if (previous && this.awaitingPackaged && this.captureWorkObserved && !snapshot.active && !snapshot.imu_active && !snapshot.busy) {
      const key = this.artifactKey(snapshot);
      // Closing the viewer may publish an idle snapshot before packaging starts.
      // Only the acquired take's new archive completes this handoff.
      if (snapshot.selected_capture !== this.awaitingCaptureId) this.awaitingPackaged = false;
      else if (artifact?.name?.endsWith(".zip") && key && key !== this.captureStartArtifact) {
        this.awaitingPackaged = false;
        this.onPackaged(snapshot);
      }
    }
  }

  artifactKey(snapshot) {
    return snapshot?.artifact?.name ? JSON.stringify([snapshot.selected_capture, snapshot.artifact.name, snapshot.artifact.bytes]) : "";
  }

  updateButtons() {
    for (const root of [this.captureRoot, this.libraryRoot]) {
      for (const button of root?.querySelectorAll("[data-native-action]") || []) {
        const action = button.dataset.nativeAction;
        button.disabled = !this.enabled(action) || (action === "capture" && (!this.captureIntent || this.intentError)) || (this.configDirty && ["capture", "checkPhone", "checkConnection"].includes(action));
        if (["cancelUpload", "retryUpload"].includes(action)) button.classList.toggle("hidden", this.snapshot?.enabled?.[action] !== true);
      }
    }
    const check = this.captureRoot?.querySelector("#native-check-primary");
    const capture = this.captureRoot?.querySelector("#native-capture-primary");
    const hasCamera = Boolean(this.snapshot?.cameras?.length);
    if (check) {
      check.classList.toggle("primary-button", !hasCamera);
      check.classList.toggle("secondary-button", hasCamera);
      check.textContent = hasCamera ? "Check phone again" : "Check phone";
    }
    if (capture) {
      capture.classList.toggle("hidden", !hasCamera);
      capture.textContent = this.snapshot?.cameras?.[this.snapshot.camera_index]?.full_walk_available === true && this.captureIntent
        ? this.captureIntent.mode === "path_refinement" ? "Capture paired path" : "Capture reconstruction walk"
        : "Open camera / test";
    }
    const note = this.captureRoot?.querySelector("#native-setup-note");
    if (note) note.textContent = this.bridge.pendingAction ? "Waiting for the phone to confirm this action. Refresh phone if status does not update."
      : this.configDirty
      ? "Setup has unsaved edits. Apply setup before checking or opening the camera."
      : !hasCamera ? "I’ll check which native camera modes this phone supports. Your saved recordings stay on the phone."
        : "Open the camera to record. If a 10-second recording check is needed, the viewer will explain it. Saved recordings open in Library with Upload beside them.";
  }
}

export default NativeCaptureUI;
