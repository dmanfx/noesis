import { escapeHtml, readableDuration, captureGuidanceMarkup } from "./native_capture.js?v=2.0.0";

export const CALIBRATION_REQUEST_SCHEMA = "roomwalk.calibration_request.v1";

export const DEFAULT_BOARD = Object.freeze({
  squares_x: 10, squares_y: 14, square_length_m: 0.018, marker_length_m: 0.0132,
  dictionary: "DICT_4X4_1000", marker_ids: Object.freeze(Array.from({ length: 70 }, (_, index) => 300 + index)),
  legacy_pattern: false,
});
const FORM_KEY = "roomwalkCalibrationForm.v1";
const ACTIVE_JOB_STATES = new Set(["queued", "pending", "running", "processing", "cancelling"]);

export function motionPrerequisite(mode, cameraId, boardConfirmed) {
  if (mode !== "imu") return "";
  if (!cameraId?.trim()) return "Choose a qualified camera-calibration reference before motion capture or processing. Keep its lens and locked focus unchanged; an uploaded take remains retained.";
  if (boardConfirmed !== true) return "Confirm the measured printed board dimensions before motion capture or processing. This is a scale check, not a request for another recording.";
  return "";
}

export function motionFocusMismatch(camera, reference) {
  const focus = camera?.focus_control, signature = reference?.binding?.signature;
  // Older snapshots/offline job lists cannot establish a match; the server's
  // full immutable capture binding remains mandatory, regardless of this hint.
  if (!focus || !signature) return "";
  const expected = signature.actual_camera2?.lens_focus_distance_diopters;
  const actual = focus.focus_distance_diopters;
  if (focus.mode !== "manual_locked" || !Number.isFinite(actual) || !Number.isFinite(expected)
      || Math.fround(actual) !== Math.fround(expected)
      || focus.physical_camera_id !== signature.actual_camera2?.active_physical_camera_id
      || focus.build_fingerprint !== signature.focus_build_fingerprint) {
    return `The phone's current lens/focus does not match the selected camera calibration${Number.isFinite(expected) ? ` (expected ${expected} diopters)` : ""}. Choose a camera calibration measured at the current locked focus before recording motion. Do not refocus an existing take or change its metadata.`;
  }
  return "";
}

export function calibrationCaptureReadiness(native, requestError = "") {
  const state = (action, label, message, disabled = false) => ({ action, label, message, disabled });
  if (!native?.available) return state("none", "Native Android recorder required", "Open this step in the RoomWalk Android app; browser capture cannot supply the native motion recording.", true);
  if (requestError) return state("review_settings", "Review calibration settings", requestError);
  if (native.configDirty) return state("phone_setup", "Apply phone setup", "Phone setup has unsaved edits. Apply them in Capture before opening the camera.");
  const snapshot = native.snapshot;
  if (!snapshot || native.bridge?.pendingAction) return state("snapshot", "Refresh phone status", "Waiting for the phone to confirm its state. Refresh status if it does not update.");
  if (snapshot.active || snapshot.imu_active || snapshot.busy) return state("none", "Phone recorder is busy", snapshot.detail || snapshot.status || "Finish the current recording, camera view or packaging operation first.", true);
  if (native.enabled("capture")) return state("capture", "", "");
  if (!snapshot.cameras?.length) return state("checkPhone", "Check phone before capture", "The phone camera list has not been loaded in this app session. Check phone first; this is not another recording or calibration.", !native.enabled("checkPhone"));
  if (!snapshot.cameras[snapshot.camera_index]) return state("phone_setup", "Choose phone camera", "Choose the same camera in Capture, apply setup, then return here. Keep the calibrated lens and focus unchanged.");
  return state("snapshot", "Refresh phone status", `${snapshot.detail || snapshot.status || "The phone has not enabled camera capture."} Refresh status, or check the camera in Capture.`);
}

// Navigation is a capture checklist, never evidence that a calibration passed.
export const CALIBRATION_STEPS = Object.freeze([
  { id: "setup", label: "Setup", title: "Fix the board; keep one camera setup", text: "Mount the board flat and FIXED. Check its printed dimensions, then keep the same lens, zoom and locked focus for both camera recordings and later walks.", action: "Check board settings" },
  { id: "noise", label: "Still · 60s", title: "Stationary sensors · 60 seconds", text: "Set the PHONE on a stable surface during the countdown. Leave it untouched until recording finishes. Keep RoomWalk open; the camera stays off.", action: "Record stationary sensors" },
  { id: "camera", label: "Camera · 60s", title: "Camera coverage · 60 seconds", text: "Leave the BOARD FIXED and move the PHONE. Cover the image centre and edges from varied angles, staying near the distance where the board is sharp. Keep focus locked; reduce movement if the board blurs.", action: "Open camera coverage capture" },
  { id: "imu", label: "Motion · 90s", title: "Camera + motion · 90 seconds", text: "Leave the BOARD FIXED and move the PHONE. Follow the on-screen prompts: start still, rotate and move in all directions, then finish still. Keep the board visible and focus locked.", action: "Open dynamic phone capture" },
  { id: "process", label: "Saved takes", title: "Process an existing recording", text: "New recordings offer Upload and Process right here. Use these controls to resume an older saved take.", action: "Choose a saved recording" },
  { id: "profile", label: "Check / reuse", title: "Check your reusable motion profile", text: "Choose the three completed jobs. Only a passing profile can be selected for future matching walks.", action: "Choose jobs for the profile check" },
  { id: "walk", label: "First walk", title: "Your first matching walk", text: "Use the same lens and locked focus. Record a 1–5 minute walk, start still, move slowly and return to the start. Upload it to verify the result.", action: "Open normal room-walk camera" },
]);

export function calibrationNextAction(reason, mode = "camera") {
  const code = String(reason || "");
  if (/openvins|backend|vio.*unavailable|executable/.test(code)) return "Restore the server’s OpenVINS validation backend, then retry the profile check with the retained jobs. A longer phone recording cannot fix a missing software backend.";
  if (/three_hours|random_walk.*(not_observable|unmeasured|heldout_unstable)|long.term/.test(code)) return "Long-term bias random walk is not established by this short session. Check whether the server reports a usable short-session candidate; do not treat this as a request for a long recording.";
  if (/stationarity|not_stationary/.test(code)) return "Repeat the ~60-second sensors-only take with the phone resting untouched on a stable surface.";
  if (/white_heldout_unstable/.test(code)) return "Repeat the stationary take. Set the phone down during the countdown, then leave it untouched until recording finishes.";
  if (/cadence|sensor_rate|sample.*gap|timestamp.*gap|short.*duration|too_short|shorter_than/.test(code)) return "Repeat the short take with RoomWalk in the foreground. Keep the phone still for the sensor step and check that both sample counts advance.";
  if (/noise_(reference|job)|imu_noise|noise_model/.test(code)) return "Upload and process the ~60-second stationary sensor recording, then select a server-verified usable noise job from this phone.";
  if (/noise_.*(binding|unit|coefficient)/.test(code)) return "Use a stationary noise job from this exact phone and sensors. Re-record the short sensor step if the binding cannot be verified.";
  if (/board.*(scale|geometry)|physical_board_scale/.test(code)) return "Measure the printed square and marker sizes, correct the board settings, confirm them, and process again. Keep the board fixed.";
  if (/focus|changing_|crop|orientation|lens|stabiliz/.test(code)) return "Check the selected physical lens, locked focus, orientation, crop and stabilization. Re-record with one unchanged setup; a different setup needs its own calibration.";
  if (/coverage|corners|board.*detect|charuco/.test(code)) return "Repeat camera coverage: leave the board fixed and move the phone so the sharp target covers the image centre, edges and corners from varied angles. Stay within the locked focus's sharp range; do not chase closer/farther views that blur the board.";
  if (/tilt/.test(code)) return "Leave the board fixed and move the phone to view it from more varied angles, without losing sharpness or visibility.";
  if (/translation|lever_arm|acceleration|visual_motion/.test(code)) return "Repeat the 60–90-second dynamic take with the board fixed. Translate the phone left/right, up/down and nearer/farther as well as rotating it; reserve a different closing motion for validation.";
  if (/time_offset|gyro_residual/.test(code)) return "Repeat the dynamic phone take with varied turns about all three axes and changing speeds, keeping the fixed board visible. The server must verify timing on held-out motion.";
  if (/reprojection|focal_instability/.test(code)) return "Check the measured board size and locked focus. Repeat varied phone views of the fixed board in brighter light, avoiding blur and extreme angles.";
  if (/row_|skew|exposure|sensor_pixel|geometry_unverified|clock|timestamp/.test(code)) return "The native capture is missing required timing or sensor geometry evidence. Retain this take and check the phone report and recorder support before retrying; more recording time alone will not repair it.";
  return mode === "noise" ? "Review the sensor report below and repeat only the affected short sensor step. Keep the original evidence."
    : "Review the fit and capture notes below. Correct the reported setup or motion issue, then repeat only the affected short take; elapsed time alone cannot qualify it.";
}

export function parseMarkerIds(value) {
  if (Array.isArray(value)) {
    if (!value.length || value.length > 1000 || value.some((id) => !Number.isSafeInteger(id) || id < 0 || id > 2_147_483_647)) {
      throw new Error("Marker IDs must be 1–1000 non-negative integers.");
    }
    if (new Set(value).size !== value.length) throw new Error("Marker IDs must be unique.");
    return [...value];
  }
  const text = String(value ?? "").trim();
  if (!text || text.length > 16_000) throw new Error("Enter the printed marker IDs, for example 300..369.");
  const ids = [];
  for (const token of text.split(/[\s,;]+/)) {
    const match = /^(\d+)(?:(?:\.\.|-)(\d+))?$/.exec(token);
    if (!match) throw new Error("Use comma-separated marker IDs or inclusive ranges such as 300..369.");
    const first = Number(match[1]);
    const last = match[2] === undefined ? first : Number(match[2]);
    if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || last < first || last > 2_147_483_647 || ids.length + last - first + 1 > 1000) {
      throw new Error("Marker ranges must be ascending and contain at most 1000 IDs.");
    }
    for (let id = first; id <= last; id += 1) ids.push(id);
  }
  return parseMarkerIds(ids);
}

export function validateBoard(values) {
  const board = {};
  for (const key of ["squares_x", "squares_y"]) {
    const value = Number(values[key]);
    if (!Number.isInteger(value) || value < 3 || value > 30) throw new Error("Board rows and columns must be whole numbers from 3 to 30.");
    board[key] = value;
  }
  for (const key of ["square_length_m", "marker_length_m"]) {
    const value = Number(values[key]);
    if (!Number.isFinite(value) || value <= 0) throw new Error("Square and marker lengths must be positive measurements in metres.");
    board[key] = value;
  }
  if (board.marker_length_m >= board.square_length_m) throw new Error("Marker length must be smaller than square length.");
  if (board.marker_length_m < 0.001 || board.square_length_m > 0.2 || Math.max(board.squares_x, board.squares_y) * board.square_length_m > 3) throw new Error("Use markers at least 1 mm, squares at most 200 mm, and a board no wider or taller than 3 metres.");
  board.dictionary = String(values.dictionary || "").trim();
  if (!/^DICT_([4567])X\1_(50|100|250|1000)$/.test(board.dictionary)) throw new Error("Choose an OpenCV 4×4, 5×5, 6×6, or 7×7 ArUco dictionary with 50, 100, 250, or 1000 markers.");
  board.marker_ids = parseMarkerIds(values.marker_ids);
  const count = Math.floor(board.squares_x * board.squares_y / 2);
  const capacity = Number(board.dictionary.split("_").at(-1));
  if (board.marker_ids.length !== count || board.marker_ids.some((id) => id >= capacity)) throw new Error(`This board needs exactly ${count} unique marker IDs below ${capacity}. Update the explicit printed IDs; they are never rewritten automatically.`);
  if (typeof values.legacy_pattern !== "boolean") throw new Error("Specify whether the printed board uses the legacy pattern.");
  board.legacy_pattern = values.legacy_pattern;
  // The server additionally validates OpenCV topology and measured geometry.
  return board;
}

export function calibrationRequest({ scan_id, mode, board, board_geometry_confirmed, camera_calibration_id, noise_calibration_id }) {
  if (typeof scan_id !== "string" || !scan_id.trim() || scan_id.length > 160) throw new Error("Choose an uploaded capture to process.");
  if (!["camera", "imu"].includes(mode)) throw new Error("Choose camera intrinsics or camera–IMU calibration.");
  if (typeof board_geometry_confirmed !== "boolean") throw new Error("Record whether the printed board geometry was checked.");
  const payload = { schema: CALIBRATION_REQUEST_SCHEMA, scan_id, mode, board: validateBoard(board), board_geometry_confirmed };
  for (const [key, id] of Object.entries({ camera_calibration_id, noise_calibration_id })) {
    if (id === undefined || id === "") continue;
    if (typeof id !== "string" || id.length > 160 || !id.trim()) throw new Error("Enter a valid measured calibration job ID.");
    payload[key] = id.trim();
  }
  return payload;
}

export function motionProfileRequest(values) {
  const request = {};
  for (const [key, label] of [["camera_calibration_id", "camera"], ["imu_calibration_id", "camera–IMU"], ["noise_calibration_id", "noise"]]) {
    const value = values?.[key];
    if (typeof value !== "string" || !value.trim() || value.length > 160) throw new Error(`Choose a completed ${label} job for the motion-profile check.`);
    request[key] = value.trim();
  }
  return request;
}

export function selectedCalibrationRequest(snapshot) {
  const request = snapshot?.calibration_request || snapshot?.selected_capture?.calibration_request;
  if (!request || typeof request !== "object" || Array.isArray(request)) return null;
  if (!Object.keys(request).length) return null; // An uninitialized native preference is not a tagged take.
  const captureId = typeof snapshot.selected_capture === "string" ? snapshot.selected_capture : snapshot.selected_capture?.id;
  if (request.capture_id && request.capture_id !== captureId) return null;
  return request;
}

export function safeReportUrl(value, origin) {
  if (typeof value !== "string" || !value || value.length > 4096) return null;
  try {
    const url = new URL(value, origin);
    return ["https:", "http:"].includes(url.protocol) && url.origin === new URL(origin).origin && !url.username && !url.password ? url.href : null;
  } catch (_) { return null; }
}

export function scanCameraSelectionMarkup(scan) {
  // Only the frame-prep worker's root result is application evidence. The
  // immutable capture.camera_calibration_selection is intent at import time.
  const result = scan?.camera_calibration_selection;
  const motion = scanMotionProfileMarkup(scan);
  if (!result || !["applied", "not_applied"].includes(result.status)) return motion;
  const applied = result.status === "applied";
  const identity = typeof result.profile_id === "string" ? result.profile_id : typeof result.camera_calibration_id === "string" ? result.camera_calibration_id : "";
  const reasons = (Array.isArray(result.reason_codes) ? result.reason_codes : []).filter((value) => typeof value === "string").slice(0, 12);
  return `<section class="camera-selection-card" data-scan-camera-selection="${result.status}"><strong>${applied ? "Camera profile applied to this scan" : "Selected camera profile not applied"}</strong>${identity ? `<p class="tool-note">${escapeHtml(identity.slice(0, 160))}</p>` : ""}${reasons.length ? `<p class="tool-note">${applied ? "Processing notes" : "Reasons"}: ${reasons.map((reason) => escapeHtml(reason.slice(0, 240))).join(" · ")}</p>` : ""}<p class="tool-note">${applied ? "The frame-preparation worker reports this profile was applied." : "The frame-preparation worker did not apply the selected profile."} Camera-profile application does not establish metric VIO acceptance.</p></section>${motion}`;
}

export function scanMotionProfileMarkup(scan) {
  const capture = scan?.capture, result = capture?.motion_profile;
  // A selection snapshot is intent, not an applied report or a VIO result.
  if (!result || !["applied", "not_applied"].includes(result.status)) return "";
  const applied = result.status === "applied" && capture.metric_vio_allowed === true;
  const reasons = (Array.isArray(result.reason_codes) ? result.reason_codes : []).filter((reason) => typeof reason === "string").slice(0, 12);
  const report = safeReportUrl(capture.motion_profile_import_url, globalThis.location?.origin);
  const retry = !applied && capture.motion_calibration_selection && scan.prepared && ["ready", "complete"].includes(scan.status) && !["queued", "running", "complete"].includes(scan.vio?.status);
  return `<section class="camera-selection-card" data-scan-motion-profile="${applied ? "applied" : "not_applied"}"><strong>${applied ? "Motion profile bound to this walk" : "Motion profile not applied · RGB remains available"}</strong>
    ${typeof result.profile_id === "string" ? `<p class="tool-note">${escapeHtml(result.profile_id.slice(0, 160))}</p>` : ""}
    <p class="tool-note">${escapeHtml(String(result.message || "Review the profile binding and retained capture report.").slice(0, 2000))}</p>
    ${reasons.length ? `<p class="tool-note">${reasons.map((reason) => escapeHtml(humanLabel(reason))).join(" · ")}</p>` : ""}
    <p class="tool-note">${escapeHtml(String(result.next_action || (applied ? "Run OpenVINS, then review tracking, resets and drift on this normal walk. Profile matching is not a room-accuracy certificate." : "Check the exact phone, lens and locked focus. Restore missing profile/backend files and recheck, or record a new matching walk. Existing imports do not switch to a newly selected profile.")).slice(0, 2000))}</p>
    ${retry ? '<button id="initiate-vio" class="secondary-button" type="button">Recheck profile and run OpenVINS</button>' : ""}
    ${report ? `<p class="tool-note"><a href="${escapeHtml(report)}" target="_blank" rel="noopener">Derived motion-profile report</a> · raw capture evidence is unchanged</p>` : ""}</section>`;
}

export async function calibrationJsonFetch(url, options = {}, { fetchImpl = globalThis.fetch?.bind(globalThis), timeoutMs = 12_000 } = {}) {
  const controller = new AbortController();
  let timeout;
  try {
    return await Promise.race([
      (async () => {
        const response = await fetchImpl(url, { ...options, signal: controller.signal });
        if (!response.ok) {
          let message = response.status === 404 && url === "/api/calibration/jobs" && options.method !== "POST"
            ? "Calibration processing is not available on this server yet. Native capture and board editing still work."
            : `Calibration request failed (HTTP ${response.status}).`;
          try {
            const error = await response.json();
            if (typeof error.detail === "string" && error.detail !== "Not Found") message = error.detail;
            else if (error.detail && typeof error.detail !== "string") message = JSON.stringify(error.detail);
          } catch (_) { /* Retain the useful HTTP error. */ }
          throw new Error(message);
        }
        return response.json();
      })(),
      new Promise((_, reject) => {
        timeout = setTimeout(() => { controller.abort(); reject(new Error("Calibration server did not respond in time. Your recordings and board settings are unchanged.")); }, timeoutMs);
      }),
    ]);
  } finally { clearTimeout(timeout); }
}

function jobId(job) { return typeof job?.id === "string" ? job.id : typeof job?.job_id === "string" ? job.job_id : ""; }
function imuCaptureId(receipt) { return receipt?.imu_capture_id || receipt?.capture_id || receipt?.id || ""; }
function pretty(value) { return JSON.stringify(value, null, 2); }

function humanLabel(value) {
  return String(value || "").slice(0, 240).replaceAll("_", " ").replace(/\b(imu|vio|rms|p95|xy)\b/gi, (word) => word.toUpperCase()).replace(/^./, (letter) => letter.toUpperCase());
}

function measured(value, unit = "", digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toLocaleString("en-US", { maximumFractionDigits: digits })}${unit ? ` ${unit}` : ""}` : null;
}

function metricCards(rows) {
  const present = rows.filter(([, value]) => value !== null && value !== undefined && value !== "").slice(0, 24);
  return present.length ? `<dl class="result-metrics">${present.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : "";
}

const QUALIFICATION_GATES = [
  ["camera_intrinsics_calibrated", "Camera intrinsics"],
  ["camera_imu_extrinsics_calibrated", "Camera–IMU extrinsics"],
  ["time_offset_calibrated", "Camera–IMU time offset"],
  ["noise_model_usable", "Short-session noise candidate"],
  ["imu_noise_calibrated", "Full IMU noise calibration (including long-term random walk)"],
  ["motion_profile_ready", "Motion profile · held-out board check"],
  ["accepted_for_metric_vio", "Metric VIO acceptance"],
];

export function calibrationOutcome(job) {
  const report = job.result && typeof job.result === "object" ? job.result : {};
  const quality = report.quality?.status;
  let title = "Qualification not reported", tone = "neutral";
  if (["failed", "cancelled"].includes(job.status) || report.status === "failed") { title = job.status === "cancelled" ? "Cancelled · evidence retained" : "Processing failed · evidence retained"; tone = "error"; }
  else if (ACTIVE_JOB_STATES.has(job.status)) title = job.status === "queued" ? "Queued for processing" : job.status === "cancelling" ? "Cancellation in progress" : "Calibration processing";
  else if (report.accepted_for_metric_vio === true) { title = "Server reports metric VIO acceptance"; tone = "success"; }
  else if (job.mode === "motion_profile" && job.status === "completed" && report.motion_profile_ready === true) { title = "Motion profile ready for matching short walks"; tone = "success"; }
  else if (job.mode === "camera" && report.camera_intrinsics_calibrated === true) { title = "Camera fit qualified · review only"; tone = "success"; }
  else if (job.mode === "noise" && report.imu_noise_calibrated === true) { title = "IMU noise qualified · review only"; tone = "success"; }
  else if (job.mode === "noise" && report.noise_model_usable === true) { title = "Short-session noise candidate · usable for verification"; tone = "success"; }
  else if (quality === "rejected") { title = "Fit rejected · retained for review"; tone = "error"; }
  else if (quality === "insufficient_evidence") { title = "Computed · more evidence needed"; tone = "warning"; }
  else if (job.status === "completed") title = "Processing complete · review qualification below";
  const gates = QUALIFICATION_GATES.map(([key, label]) => ({ key, label, value: report[key] === true ? key === "noise_model_usable" ? "Usable candidate" : "Qualified" : report[key] === false ? key === "accepted_for_metric_vio" ? "Not accepted" : "Not established" : "Not reported", tone: report[key] === true ? "success" : report[key] === false ? "warning" : "neutral" }));
  const reasons = [...new Set([...(Array.isArray(report.reason_codes) ? report.reason_codes : []), ...(Array.isArray(report.quality?.reason_codes) ? report.quality.reason_codes : [])])].filter((value) => typeof value === "string").slice(0, 24);
  return { report, quality, title, tone, gates, reasons };
}

export function calibrationSummaryMarkup(job) {
  const outcome = calibrationOutcome(job), report = outcome.report;
  const camera = report.camera_result && typeof report.camera_result === "object" ? report.camera_result : {};
  const quality = camera.quality && typeof camera.quality === "object" ? camera.quality : report.quality || {};
  const percent = typeof job.progress === "number" && Number.isFinite(job.progress) ? Math.max(0, Math.min(100, job.progress * 100)) : null;
  const rows = [["Model", typeof camera.model === "string" ? humanLabel(camera.model) : null],
    ["Native resolution", Array.isArray(camera.resolution) ? `${camera.resolution.slice(0, 2).join(" × ")} px` : null],
    ["Training RMS", measured(camera.training_rms_px, "px")],
    ["Held-out RMS", measured(quality.heldout?.radial_px?.rms, "px")],
    ["Held-out P95", measured(quality.heldout?.radial_px?.p95, "px")],
    ["Reverse held-out RMS", measured(quality.reverse_heldout?.radial_px?.rms, "px")],
    ["Tilt span", measured(quality.tilt_span_deg, "°")],
    ["Training / held-out views", Array.isArray(camera.training_frame_indices) && Array.isArray(camera.holdout_frame_indices) ? `${camera.training_frame_indices.length} / ${camera.holdout_frame_indices.length}` : null],
    ["Solver status", typeof report.solver_status === "string" ? humanLabel(report.solver_status) : null]];
  if (Array.isArray(camera.K) && camera.K.length === 3) {
    rows.push(["Focal length fx / fy", `${measured(camera.K[0]?.[0]) || "—"} / ${measured(camera.K[1]?.[1]) || "—"} px`], ["Principal point cx / cy", `${measured(camera.K[0]?.[2]) || "—"} / ${measured(camera.K[1]?.[2]) || "—"} px`]);
  }
  if (Array.isArray(camera.D)) rows.push(["Distortion coefficients", camera.D.flat().slice(0, 14).map((value) => measured(value, "", 6) || "—").join(", ")]);
  if (Array.isArray(quality.coverage_fraction_xy)) rows.push(["Image coverage X / Y", quality.coverage_fraction_xy.slice(0, 2).map((value) => measured(value * 100, "%", 1) || "—").join(" / ")]);
  if (typeof quality.reverse_focal_fraction === "number") rows.push(["Reverse-split focal change", measured(quality.reverse_focal_fraction * 100, "%")]);
  const estimate = report.calibration_estimate;
  for (const [key, label] of [["cam_time_offset_ns", "Solver camera time offset"], ["imu_to_camera_offset_ns", "IMU → camera timestamp offset"]]) {
    if (typeof estimate?.[key] === "number") rows.push([label, measured(estimate[key] / 1e6, "ms", 6)]);
  }
  if (Array.isArray(estimate?.T_imu_camera) && estimate.T_imu_camera.length === 4) rows.push(["Estimated T_imu_camera translation", estimate.T_imu_camera.slice(0, 3).map((row) => measured(row?.[3], "m", 6) || "—").join(" / ")]);
  const noiseRows = Object.entries(report.noise || {}).slice(0, 4).map(([key, value]) => {
    const provenance = report.noise_provenance?.[key];
    const prior = provenance?.measured === false || provenance?.kind === "model_prior" || (key.endsWith("random_walk") && report.imu_noise_calibrated === false);
    const basis = prior ? "model prior, not measured" : provenance?.measured === true || report.imu_noise_calibrated === true ? "measured" : "provenance not reported";
    return [`${humanLabel(key)} · ${basis}`, typeof value === "number" && Number.isFinite(value) ? `${value.toPrecision(4)} ${typeof report.units?.[key] === "string" ? report.units[key] : "(units not reported)"}` : null];
  });
  const streams = Object.entries(report.streams || {}).filter(([kind, value]) => ["accelerometer", "gyroscope"].includes(kind) && value && typeof value === "object");
  const streamCards = streams.map(([kind, stream]) => `<section class="noise-stream-summary"><h4>${escapeHtml(humanLabel(kind))}</h4>${metricCards([["Recorded duration", typeof stream.duration_s === "number" ? readableDuration(stream.duration_s) : null], ["Sample rate", measured(stream.rate_hz, "Hz")], ["Samples", measured(stream.sample_count, "", 0)], ["Maximum timestamp gap", measured(stream.maximum_gap_s, "s", 6)]])}</section>`).join("");
  const fitRows = streams.flatMap(([kind, stream]) => (Array.isArray(stream.axes) ? stream.axes : []).slice(0, 3).flatMap((axis) => ["white", "random_walk"].map((regime) => {
    const fit = axis.fits?.[regime];
    return `<tr><th scope="row">${escapeHtml(humanLabel(kind))} ${escapeHtml(axis.axis)} · ${regime === "white" ? "white noise" : "random walk"}${typeof fit?.coefficient === "number" ? `<small>Coefficient ${escapeHtml(fit.coefficient.toPrecision(4))}${typeof fit.heldout_ratio === "number" ? ` · held-out ratio ${escapeHtml(measured(fit.heldout_ratio))}` : ""}</small>` : ""}</th><td>${fit?.heldout_passed === true ? "Held-out check passed" : fit ? "Held-out check not passed" : "Not observable"}</td></tr>`;
  })));
  const errors = [job.error, report.error].filter((value) => typeof value === "string" && value);
  return `<div class="result-heading"><div><span class="eyebrow">${escapeHtml(humanLabel(job.mode))} · ${escapeHtml(job.status || "Status not reported")}</span><h3>${escapeHtml(outcome.title)}</h3></div><span class="result-badge ${outcome.tone}">${escapeHtml(humanLabel(outcome.quality || job.status || "Unknown"))}</span></div>
    ${job.message && ACTIVE_JOB_STATES.has(job.status) ? `<p class="tool-note">${escapeHtml(String(job.message).slice(0, 2000))}</p>` : ""}
    ${ACTIVE_JOB_STATES.has(job.status) ? `<progress class="transfer-progress" max="100" ${percent === null ? "" : `value="${percent}"`} aria-label="Calibration processing progress"></progress>${percent === null ? "" : `<p class="tool-note">${Math.floor(percent)}% · processing progress, not a quality score</p>`}` : ""}
    ${errors.length ? `<div class="result-error">${[...new Set(errors)].map((error) => `<p>${escapeHtml(error.slice(0, 4000))}</p>`).join("")}</div>` : ""}
    ${outcome.reasons.length ? `<section class="result-reasons"><h4>Next action</h4>${[...new Set(outcome.reasons.map((reason) => calibrationNextAction(reason, job.mode)))].map((action) => `<p>${escapeHtml(action)}</p>`).join("")}</section>` : ""}
    <details class="tool-details" data-calibration-measurements><summary>Measurements and checks</summary>
    <div class="result-table-wrap"><table class="result-gates"><caption>Reported qualification gates</caption><thead><tr><th scope="col">Measurement</th><th scope="col">Outcome</th></tr></thead><tbody>${outcome.gates.map((gate) => `<tr><th scope="row">${gate.label}</th><td><span class="result-badge ${gate.tone}">${gate.value}</span></td></tr>`).join("")}</tbody></table></div>
    <p class="tool-note">“Not established” means this job does not qualify that measurement. A camera-only or noise-only job does not qualify camera–IMU alignment. This page does not change live Noesis settings.</p>
    ${report.noise_model_usable === true && report.imu_noise_calibrated !== true ? '<p class="tool-status">Usable short-session candidate. Long-term bias random walk remains unmeasured; full IMU noise calibration is not established. Continue with camera–IMU verification and an independent normal walk.</p>' : ""}
    ${job.mode === "motion_profile" ? '<p class="tool-status">This checks the fixed profile with OpenVINS on held-out board motion. It is not a five-minute room-accuracy certificate. Review the first normal walk separately; selection applies only to future matching native imports.</p>' : ""}
    ${outcome.reasons.length ? `<section class="result-reasons"><h4>What to do next</h4><ul>${outcome.reasons.map((reason) => `<li><strong>${escapeHtml(humanLabel(reason))}</strong><p>${escapeHtml(calibrationNextAction(reason, job.mode))}</p></li>`).join("")}</ul></section>` : quality.status === "insufficient_evidence" ? `<section class="result-reasons"><h4>What to do next</h4><p>${escapeHtml(calibrationNextAction("", job.mode))}</p></section>` : ""}
    ${rows.some(([, value]) => value !== null) ? `<section><h4>Model and fit summary</h4>${metricCards(rows)}</section>` : ""}
    ${noiseRows.length ? `<section><h4>Noise model coefficients</h4>${metricCards(noiseRows)}</section>` : ""}${streamCards}
    ${fitRows.length ? `<details class="tool-details"><summary>Noise fit checks by sensor and axis</summary><div class="result-table-wrap"><table class="result-gates"><thead><tr><th scope="col">Fit</th><th scope="col">Held-out evidence</th></tr></thead><tbody>${fitRows.join("")}</tbody></table></div></details>` : ""}
    ${Array.isArray(report.limitations) && report.limitations.length ? `<details class="tool-details"><summary>Measurement limitations</summary><ul class="result-limitations">${report.limitations.slice(0, 12).filter((value) => typeof value === "string").map((value) => `<li>${escapeHtml(value.slice(0, 1500))}</li>`).join("")}</ul></details>` : ""}</details>`;
}

export function calibrationJobCards(jobs, selectedId) {
  return jobs.filter((job) => jobId(job)).slice(0, 100).map((job) => `<button type="button" data-calibration-job="${escapeHtml(jobId(job))}" aria-pressed="${jobId(job) === selectedId}"><strong>${escapeHtml(job.mode === "camera" ? "Camera intrinsics" : job.mode === "imu" ? "Camera–IMU" : job.mode === "noise" ? "Stationary IMU noise" : job.mode === "motion_profile" ? "Motion-profile check" : "Calibration")} · ${escapeHtml(job.status || "Status not reported")}</strong><small>${escapeHtml(job.scan_id || job.imu_capture_id || jobId(job))}</small></button>`).join("") || '<p class="tool-note">No calibration jobs have been reported.</p>';
}

export class CalibrationUI {
  constructor({ root, nativeCapture = null, platform = globalThis, fetchImpl, onOpenScan = () => {} } = {}) {
    this.root = root;
    this.nativeCapture = nativeCapture;
    this.platform = platform;
    this.fetchImpl = fetchImpl || platform.fetch?.bind(platform);
    this.onOpenScan = onOpenScan;
    this.scans = [];
    this.jobs = [];
    this.selectedJobId = "";
    this.selectedJob = null;
    this.visible = false;
    this.loading = false;
    this.jobsLoaded = false;
    this.jobsError = "";
    this.recoveryMode = null;
    this.pendingResume = null;
    this.motionSourceChoices = {};
    this.submitting = false;
    this.pollTimer = null;
    this.pollFailures = 0;
    this.detailRequest = 0;
    this.nativeRequestKey = "";
    this.nativeOutcomeKey = "";
    this.hasNativeCalibrationRequest = false;
    this.captureRequestError = "";
    this.imuRecordings = [];
    this.noiseSubmitting = false;
    this.noiseLoading = false;
    this.cameraSelection = null;
    this.cameraSelectionLoaded = false;
    this.cameraSelecting = false;
    this.cameraSelectionRevision = 0;
    this.guideStep = "setup";
    this.motionSelection = null;
    this.motionSelectionLoaded = false;
    this.motionSelectionRevision = 0;
    this.motionSelecting = false;
    this.motionSubmitting = false;
    this.workflow = null;
    this.sensorRunActive = false;
    this.motionProfileAvailable = undefined;
    this.renderShell();
    this.restoreForm();
    this.bind();
    this.updateNative();
    this.renderGuide();
  }

  element(selector) { return this.root?.querySelector(selector); }

  motionPrerequisite(mode = this.element("#calibration-mode")?.value, forCapture = false) {
    const id = this.element("#calibration-camera-id")?.value;
    const basic = motionPrerequisite(mode, id, this.element("#board-confirmed")?.checked);
    if (basic || mode !== "imu" || !forCapture) return basic;
    const snapshot = this.nativeCapture?.snapshot;
    const job = this.jobs.find((row) => jobId(row) === id && row.mode === "camera");
    return motionFocusMismatch(snapshot?.cameras?.[snapshot.camera_index], job?.result?.camera_result);
  }

  reviewMotionPrerequisite() {
    const selector = this.captureRequestError || (this.element("#calibration-camera-id")?.value.trim() && !this.element("#board-confirmed")?.checked) ? "#calibration-board-settings" : "#calibration-camera-job";
    this.openSection(selector);
  }

  stageLabel(mode) { return { noise: "Stationary sensors", camera: "Camera", imu: "Motion" }[mode]; }

  missingProfileSources() {
    return ["noise", "camera", "imu"].filter((mode) => !this.jobs.some((job) => job.mode === mode && job.status === "completed"));
  }

  renderProfilePrerequisites() {
    const root = this.element("#calibration-prerequisites");
    if (!root) return;
    root.classList.toggle("hidden", this.guideStep !== "profile");
    const missing = this.missingProfileSources();
    this.element("#motion-profile-form").classList.toggle("hidden", !this.jobsLoaded || missing.length > 0);
    const explanation = this.element("#motion-profile-missing");
    explanation.classList.toggle("hidden", this.jobsLoaded && missing.length === 0);
    explanation.textContent = !this.jobsLoaded ? this.jobsError || "Loading calibration results…" : `${missing.map((mode) => this.stageLabel(mode)).join(" and ")} ${missing.length > 1 ? "recordings need" : "recording needs"} uploading and processing before this check can run.`;
    const resume = this.element("#motion-continue-missing");
    resume.classList.toggle("hidden", !this.jobsLoaded || !missing.length);
    resume.textContent = missing.length ? `Continue ${this.stageLabel(missing[0]).toLowerCase()} setup` : "Continue setup";
    this.element("#motion-results-refresh").disabled = this.loading;
    if (!this.jobsLoaded) {
      root.textContent = this.jobsError || "Loading calibration results…";
      return;
    }
    root.innerHTML = ["noise", "camera", "imu"].map((mode) => {
      const count = this.jobs.filter((job) => job.mode === mode && job.status === "completed").length;
      return `<p><strong>${this.stageLabel(mode)}</strong> · ${count ? `${count} processed result${count === 1 ? "" : "s"} available` : "No processed result yet"}</p>`;
    }).join("") + (missing.length ? "<p>First upload and process the missing calibration takes. Saved recordings can be resumed below; do not repeat your room walk.</p>" : "<p>Choose the three results below, then press Check motion profile. Processing completion alone is not a quality pass.</p>");
  }

  openRecovery(mode) {
    if (!["noise", "camera", "imu"].includes(mode)) return;
    this.recoveryMode = mode;
    this.setGuideStep(mode, { focus: false });
    this.message("");
    this.renderRecovery();
    this.openSection("#calibration-recovery");
  }

  renderRecovery() {
    const root = this.element("#calibration-recovery");
    if (!root) return;
    const mode = this.recoveryMode;
    root.classList.toggle("hidden", !mode || mode !== this.guideStep);
    if (!mode) return;
    const label = this.stageLabel(mode);
    this.element("#calibration-recovery-title").textContent = `Continue ${label.toLowerCase()} setup`;
    const native = this.nativeCapture?.snapshot;
    const phone = (native?.saved || []).filter((take) => !take.metadata_error && (mode === "noise" ? take.kind === "imu_diagnostic" : take.calibration_mode === mode && take.short_test !== true));
    const uploaded = mode === "noise" ? this.imuRecordings.map((take) => ({ id: imuCaptureId(take), name: imuCaptureId(take), status: "stored" }))
      : this.scans.filter((scan) => scan.capture?.calibration_request?.mode === mode && scan.capture.calibration_request.short_test !== true);
    this.element("#calibration-recovery-message").textContent = phone.length || uploaded.length
      ? "Choose the recording you already made. Nothing uploads or processes until you press its next action."
      : `No ${label.toLowerCase()} recording is listed yet. Refresh recordings to check the phone and server before recording again. Ten-second timing tests are not calibration takes.`;
    this.element("#calibration-recovery-list").innerHTML = [
      ...phone.map((take) => `<button type="button" class="secondary-button" data-recovery-phone="${escapeHtml(take.id)}" ${this.nativeCapture?.enabled("selectSaved") ? "" : "disabled"}>${escapeHtml(label)} · saved on phone<br><small>${escapeHtml(take.name || take.id)} · ${take.upload_ready ? "Ready to upload" : "Needs packaging"}</small></button>`),
      ...uploaded.map((take) => `<button type="button" class="secondary-button" data-recovery-server="${escapeHtml(take.id)}">${escapeHtml(label)} · on server<br><small>${escapeHtml(take.name || take.id)} · ${escapeHtml(take.status)}</small></button>`),
    ].join("");
    this.element("#calibration-recovery-record").textContent = `Record a new ${label.toLowerCase()} take`;
    this.element("#calibration-recovery-record").disabled = !this.nativeCapture?.enabled(mode === "noise" ? "startImu" : "capture");
  }

  resumeServerTake(id) {
    const mode = this.recoveryMode;
    const source = mode === "noise" ? this.imuRecordings.find((take) => imuCaptureId(take) === id) : this.scans.find((scan) => scan.id === id && scan.capture?.calibration_request?.mode === mode && scan.capture.calibration_request.short_test !== true);
    if (!source) return;
    this.workflow = { mode, phase: "uploaded", captureId: mode === "noise" ? id : source.capture?.capture_id, scanId: mode === "noise" ? undefined : id };
    this.recoveryMode = null;
    if (mode !== "noise") { this.element("#calibration-scan").value = id; this.loadSelectedScanRequest(); }
    this.setGuideStep(mode, { focus: false });
    this.message("");
    this.openSection("#calibration-guide-content");
  }

  focusProgress() {
    const target = this.element("#calibration-run");
    target?.classList.remove("hidden");
    target?.focus({ preventScroll: true });
    target?.scrollIntoView({ block: "start", behavior: "instant" });
  }

  openSection(selector) {
    const target = this.element(selector);
    for (let parent = target; parent && parent !== this.root; parent = parent.parentElement) if (parent.tagName === "DETAILS") parent.open = true;
    target?.scrollIntoView({ block: "start", behavior: "instant" });
    target?.focus({ preventScroll: true });
  }

  handlePackaged(snapshot) {
    const mode = snapshot.imu_telemetry ? "noise" : snapshot.calibration_request?.mode;
    if (!this.visible || !["noise", "camera", "imu"].includes(mode)) return false;
    const interrupted = snapshot.capture_outcome?.capture_id === snapshot.selected_capture && snapshot.capture_outcome.partial === true;
    if (mode !== "noise" && interrupted) {
      this.recoveryMode = null;
      this.workflow = { mode, captureId: snapshot.selected_capture, phase: "capture_interrupted" };
      this.setGuideStep(mode, { focus: false });
      this.message(snapshot.capture_outcome.message || "Recording stopped early. The original is retained in Library; check the camera before another take.", true);
      this.openSection("#calibration-guide-content");
      return true;
    }
    if (mode !== "noise" && snapshot.capture_short_test === true) {
      this.workflow = null;
      this.setGuideStep(mode, { focus: false });
      const passed = snapshot.cameras?.[snapshot.camera_index]?.full_walk_available === true;
      this.message(passed ? "Timing test passed. Reopen the camera and record this step; keep focus locked." : "The timing test did not enable recording. Keep this take for diagnosis; do not change focus or repeat the calibration yet.", !passed);
      this.openSection("#calibration-guide-content");
      return true;
    }
    this.recoveryMode = null;
    this.workflow = { mode, captureId: snapshot.selected_capture, phase: "saved" };
    this.setGuideStep(mode, { focus: false });
    this.message("");
    this.openSection("#calibration-guide-content");
    return true;
  }

  async processWorkflow() {
    const flow = this.workflow;
    if (!flow) return;
    if (flow.mode === "noise") {
      await this.refreshImuRecordings();
      this.element("#noise-recording").value = flow.captureId;
      await this.submitNoise();
    } else {
      this.element("#calibration-scan").value = flow.scanId || "";
      this.loadSelectedScanRequest();
      await this.submit();
    }
  }

  setGuideStep(id, { focus = true } = {}) {
    if (!CALIBRATION_STEPS.some((step) => step.id === id)) return;
    this.guideStep = id;
    if (["camera", "imu"].includes(id)) {
      this.element("#calibration-mode").value = id;
      this.saveForm();
      this.updateNative();
    }
    this.renderGuide();
    if (id === "profile" && this.visible && !this.loading) void this.refreshJobs();
    if (focus) this.element("#calibration-guide-content")?.focus();
  }

  renderGuide() {
    const index = CALIBRATION_STEPS.findIndex((step) => step.id === this.guideStep);
    const step = CALIBRATION_STEPS[index];
    this.element("#calibration-job-detail")?.classList.toggle("hidden", Boolean(this.selectedJob && this.selectedJob.mode !== this.guideStep && !["process", "profile"].includes(this.guideStep)));
    const content = this.element("#calibration-guide-content");
    if (!content) return;
    const flow = this.workflow?.mode === step.id ? this.workflow : null;
    const instruction = {
      saved: "Recording saved on this phone. Upload it to continue.",
      needs_package: "The original recording is retained. Package it for upload; you do not need to record it again.",
      packaging: "Packaging the retained recording. The upload action will appear here when ready.",
      uploading: "Uploading your saved recording. You can leave the phone resting; no more movement is needed.",
      uploaded: "Recording received. Process it to check this step.",
      processing: "Checking your recording. The result and next action will appear here.",
      retry: "This recording needs attention. Follow the guidance below before recording again; the original is retained.",
      capture_interrupted: "The camera stopped before this take finished. Your partial recording is safe in Library; it is not a completed calibration take.",
      passed: "This step is ready. Continue to the next step when you are ready.",
    }[flow?.phase] || step.text;
    content.innerHTML = `<h3>${escapeHtml(step.title)}</h3><p>${escapeHtml(instruction)}</p>`;
    const stepSummary = this.element("#calibration-step-summary");
    if (stepSummary) stepSummary.textContent = `Step ${index + 1} of ${CALIBRATION_STEPS.length} · ${step.label}`;
    for (const button of this.root.querySelectorAll("[data-calibration-step]")) button.setAttribute("aria-current", button.dataset.calibrationStep === step.id ? "step" : "false");
    const action = this.element("#calibration-guide-action");
    action.textContent = step.action;
    const nativeAction = step.id === "noise" ? "startImu" : ["camera", "imu", "walk"].includes(step.id) ? "capture" : null;
    action.disabled = Boolean(nativeAction && (!this.nativeCapture?.enabled(nativeAction) || (nativeAction === "capture" && (this.nativeCapture.configDirty || Boolean(this.captureRequestError)))));
    if (nativeAction && !this.nativeCapture?.available) action.textContent = "Native Android recorder required";
    const capturePrerequisite = nativeAction === "capture" && (!flow || ["retry", "capture_interrupted"].includes(flow.phase))
      ? calibrationCaptureReadiness(this.nativeCapture, this.captureRequestError || this.motionPrerequisite(step.id, true)) : null;
    if (capturePrerequisite) {
      action.disabled = capturePrerequisite.disabled;
      if (capturePrerequisite.label) action.textContent = capturePrerequisite.label;
    }
    const readiness = this.element("#calibration-capture-readiness");
    if (readiness) {
      readiness.textContent = capturePrerequisite?.message || "";
      readiness.classList.toggle("hidden", !capturePrerequisite?.message);
    }
    if (step.id === "noise" && this.nativeCapture?.snapshot?.imu_preparing) action.textContent = "Settling…";
    else if (step.id === "noise" && this.nativeCapture?.snapshot?.imu_active) action.textContent = "Recording…";
    if (flow?.phase === "saved") { action.textContent = "Upload recording"; action.disabled = !this.nativeCapture?.enabled("upload") || this.nativeCapture.snapshot?.selected_capture !== flow.captureId; }
    if (flow?.phase === "needs_package") { action.textContent = "Package saved recording"; action.disabled = !this.nativeCapture?.enabled("repackage") || this.nativeCapture.snapshot?.selected_capture !== flow.captureId; }
    if (flow?.phase === "packaging") { action.textContent = "Packaging…"; action.disabled = true; }
    if (flow?.phase === "uploading") { action.textContent = "Uploading…"; action.disabled = true; }
    if (flow?.phase === "uploaded") {
      const scan = this.scans.find((scan) => scan.id === flow.scanId);
      action.textContent = scan?.status === "importing_capture" ? "Validating upload…" : "Process recording";
      action.disabled = Boolean(this.captureRequestError && flow.mode !== "noise") || (flow.mode !== "noise" && (!scan || ["importing_capture", "import_failed"].includes(scan.status)));
      if (scan?.status === "import_failed") this.message(scan.error || "The server could not validate this recording. Your original is retained; review the import error in Library.", true);
    }
    if (flow?.phase === "processing") { action.textContent = "Processing…"; action.disabled = true; }
    if (flow?.phase === "processing_failed") { action.textContent = "Review processing settings"; action.disabled = false; }
    if (flow?.phase === "retry" && !capturePrerequisite?.label) { action.textContent = "Record again"; }
    if (flow?.phase === "capture_interrupted" && !capturePrerequisite?.label) { action.textContent = "Reopen camera to check lens"; }
    if (flow?.phase === "passed") { action.textContent = `Next: ${flow.mode === "noise" ? "Camera" : flow.mode === "camera" ? "Motion" : "Check / reuse"}`; action.disabled = false; }
    if (step.id === "profile") {
      const missing = this.missingProfileSources();
      action.textContent = !this.jobsLoaded ? this.jobsError ? "Refresh calibration results" : "Loading calibration results…" : missing.length ? `Continue ${this.stageLabel(missing[0]).toLowerCase()} setup` : this.element("#motion-profile-check").disabled ? "Choose results and check profile" : "Check motion profile";
      action.disabled = this.loading && !this.jobsLoaded;
      content.innerHTML = `<h3>Check and reuse your calibration</h3><p>${missing.length ? "Some calibration recordings still need uploading or processing." : "Check the combined camera and sensor results before reusing them."}</p>`;
    }
    if (step.id === "walk" && !this.motionSelection) {
      content.innerHTML = "<h3>Room walk · no motion profile selected</h3><p>You can record for RGB reconstruction. Calibrated motion is not ready; Check / reuse lists the remaining setup.</p>";
      if (this.nativeCapture?.available && !capturePrerequisite?.label) action.textContent = "Record walk for RGB reconstruction";
    }
    action.classList.toggle("hidden", this.recoveryMode === step.id);
    const next = this.element("#calibration-guide-next");
    next.textContent = index < CALIBRATION_STEPS.length - 1 ? `View ${CALIBRATION_STEPS[index + 1].label}` : "Back to setup";
    next.classList.toggle("hidden", Boolean(flow || this.recoveryMode === step.id || step.id === "profile" || this.sensorRunActive || this.nativeCapture?.snapshot?.active));
    this.renderProfilePrerequisites();
    this.renderRecovery();
  }

  openGuideAction() {
    const id = this.guideStep;
    const flow = this.workflow?.mode === id ? this.workflow : null;
    if (flow?.phase === "needs_package") { if (this.nativeCapture?.perform("repackage")) { flow.phase = "packaging"; this.renderGuide(); } return; }
    if (flow?.phase === "saved") { if (this.nativeCapture?.perform("upload")) { flow.phase = "uploading"; this.renderGuide(); } return; }
    if (flow?.phase === "uploaded") { void this.processWorkflow(); return; }
    if (flow?.phase === "processing_failed") { this.openSection("#calibration-settings"); this.message("The saved take is retained. Correct the reported processing prerequisites, then use Process selected capture. Do not record or upload again just to retry processing.", true); return; }
    if (flow?.phase === "passed") {
      if (flow.mode === "noise") { this.element("#calibration-noise-id").value = flow.jobId; this.element("#motion-noise-job").value = flow.jobId; }
      if (flow.mode === "camera") { this.element("#calibration-camera-id").value = flow.jobId; this.element("#motion-camera-job").value = flow.jobId; }
      if (flow.mode === "imu") this.element("#motion-imu-job").value = flow.jobId;
      this.saveForm(); this.workflow = null; this.setGuideStep(flow.mode === "noise" ? "camera" : flow.mode === "camera" ? "imu" : "profile");
      if (flow.mode === "imu") this.openSection("#calibration-vio-profile");
      return;
    }
    if (["retry", "capture_interrupted"].includes(flow?.phase)) this.workflow = null;
    if (id === "profile") {
      if (!this.jobsLoaded) { void this.refreshJobs(); return; }
      const missing = this.missingProfileSources();
      if (missing.length) { this.openRecovery(missing[0]); return; }
      if (!this.element("#motion-profile-check").disabled) { void this.submitMotionProfile(); return; }
      this.openSection("#calibration-vio-profile");
      return;
    }
    if (id === "noise") { this.element("#noise-start").click(); return; }
    if (["camera", "imu", "walk"].includes(id)) {
      const prerequisite = calibrationCaptureReadiness(this.nativeCapture, this.captureRequestError || this.motionPrerequisite(id, true));
      if (prerequisite.action !== "capture") {
        if (prerequisite.disabled) { this.message(prerequisite.message, true); return; }
        if (["snapshot", "checkPhone"].includes(prerequisite.action)) this.nativeCapture.perform(prerequisite.action);
        else if (prerequisite.action === "review_settings") this.reviewMotionPrerequisite();
        else if (prerequisite.action === "phone_setup") {
          const doc = this.root.ownerDocument;
          doc.querySelector("#tab-capture")?.click();
          const setup = doc.querySelector("#native-setup");
          if (setup) setup.open = true;
          doc.querySelector("#native-camera")?.focus();
        }
        this.renderGuide();
        return;
      }
    }
    if (["camera", "imu"].includes(id)) {
      this.element("#calibration-mode").value = id;
      this.updateNative();
      this.element("#calibration-capture").click();
      return;
    }
    if (id === "walk") {
      if (this.nativeCapture?.capture("walk")) this.message("Opening a normal room walk. Keep the same lens and locked focus. Upload it and check profile matching and independent VIO validation before treating the result as metric.");
      return;
    }
    const target = this.element(id === "setup" ? "#calibration-board-settings" : id === "process" ? "#calibration-scan" : "#calibration-vio-profile");
    this.openSection(`#${target.id}`);
    if (id === "setup") this.element("#board-square-length").focus();
    else target?.focus();
  }

  renderShell() {
    if (!this.root) return;
    this.root.innerHTML = `
      <div class="tool-heading"><div><span class="eyebrow">Optional setup</span><h2>Setup</h2><p>Reconstruction does not require a calibration job. Use this secondary workspace only for supported basic camera/IMU calibration, retained sensor diagnostics, and legacy evidence review.</p></div></div>
      <details id="calibration-secondary" class="setup-secondary tool-details">
      <summary>Basic camera / IMU calibration and sensor diagnostics</summary>
      <div class="setup-secondary-content">
      <section id="calibration-run" class="calibration-run hidden" tabindex="-1" aria-label="Current recording">
        <div id="calibration-guide-progress" role="status"></div>
        <button id="noise-stop-top" type="button" class="ghost-button hidden">Stop and retain partial</button>
      </section>
      <section class="calibration-guide" aria-label="Guided short-session calibration">
        <details id="calibration-step-picker"><summary id="calibration-step-summary">Step 1 of 7 · Setup</summary><ol class="calibration-steps">${CALIBRATION_STEPS.map((step, index) => `<li><button type="button" data-calibration-step="${step.id}" aria-current="${index === 0 ? "step" : "false"}"><span>${index + 1}</span>${step.label}</button></li>`).join("")}</ol></details>
        <div id="calibration-guide-content" tabindex="-1"></div>
        <p id="calibration-message" class="tool-status hidden" role="status"></p>
        <div id="calibration-prerequisites" class="hidden" role="status"></div>
        <section id="calibration-recovery" class="hidden" tabindex="-1"><h3 id="calibration-recovery-title"></h3><p id="calibration-recovery-message"></p><div id="calibration-recovery-list" class="calibration-recovery-list"></div><div class="tool-actions"><button id="calibration-recovery-refresh" type="button" class="secondary-button">Refresh recordings</button><button id="calibration-recovery-record" type="button" class="ghost-button">Record a new take</button></div></section>
        <div id="calibration-job-detail" class="calibration-selected-job" tabindex="-1"></div>
        <p id="calibration-capture-readiness" class="tool-note hidden" role="status"></p>
        <div class="tool-actions"><button id="calibration-guide-action" type="button" class="primary-button">Check board settings</button><button id="calibration-guide-next" type="button" class="secondary-button">Next: Still · ~60s</button></div>
      </section>
      <p id="calibration-capabilities" class="tool-note hidden"></p>
      <details id="calibration-settings" class="tool-details"><summary>Capture settings and individual processing</summary>
      <form id="calibration-form">
        <div class="tool-grid">
          <label class="tool-field"><span>Calibration mode</span><select id="calibration-mode" class="text-input"><option value="camera">Camera intrinsics</option><option value="imu">Camera + IMU (extrinsics and timing)</option></select></label>
          <label class="tool-field"><span>Uploaded capture to process</span><select id="calibration-scan" class="text-input"><option value="">Upload or select a capture…</option></select><small>Recording locally and processing on the server are separate actions.</small></label>
          <label class="tool-field tool-wide"><span>Qualified camera-intrinsics reference · required for motion</span><select id="calibration-camera-job" class="text-input"><option value="">Choose a qualified camera job or enter its ID below…</option></select><small>Keep its lens and locked focus unchanged. Completion alone is not qualification; the server checks the exact capture configuration.</small></label>
          <details id="calibration-camera-advanced" class="tool-details tool-wide"><summary>Enter a camera-calibration ID manually</summary><label class="tool-field"><span>Measured camera-calibration ID</span><input id="calibration-camera-id" class="text-input" maxlength="160" autocomplete="off" placeholder="Known camera-calibration job ID" /></label></details>
        </div>
        <div class="tool-actions"><button id="calibration-capture" type="button" class="secondary-button hidden" disabled>Open calibration camera</button><button id="calibration-test" type="button" class="ghost-button hidden" disabled>Preview / 10-second test</button><button id="calibration-process" type="submit" class="primary-button" disabled>Process selected capture</button></div>
        <p id="calibration-capture-note" class="tool-note">In a browser, import the native RoomWalk camera + IMU bundle from Capture, then select it here. The Android app records the target-board session with the same native camera and sensors used for room walks.</p>
        <details id="calibration-board-settings" class="tool-details"><summary>ChArUco board · edit settings<small id="calibration-board-summary">10 × 14 · 0.018 m / 0.0132 m · IDs 300–369 · nonlegacy</small></summary>
        <p class="tool-note">Square size comes first, then marker size. Check the actual printed target; changing the form does not change the source recording.</p>
        <div class="tool-grid">
          <label class="tool-field"><span>Board columns (squares X)</span><input id="board-squares-x" class="text-input" type="number" min="3" max="30" step="1" value="10" required /></label>
          <label class="tool-field"><span>Board rows (squares Y)</span><input id="board-squares-y" class="text-input" type="number" min="3" max="30" step="1" value="14" required /></label>
          <label class="tool-field"><span>Square length (metres)</span><input id="board-square-length" class="text-input" type="number" min="0.001" max="0.2" step="any" value="0.018" required /></label>
          <label class="tool-field"><span>Marker length (metres)</span><input id="board-marker-length" class="text-input" type="number" min="0.001" max="0.2" step="any" value="0.0132" required /></label>
          <label class="tool-field tool-wide"><span>ArUco dictionary</span><input id="board-dictionary" class="text-input" value="DICT_4X4_1000" maxlength="69" spellcheck="false" required /></label>
          <label class="tool-field tool-wide"><span>Printed marker IDs (in board order)</span><textarea id="board-marker-ids" class="text-input" maxlength="16000" spellcheck="false" required>300..369</textarea><small>Inclusive ranges or comma-separated IDs. Changing dimensions does not silently rewrite the printed IDs.</small></label>
        </div>
        <label class="tool-checkbox"><input id="board-legacy" type="checkbox" /><span>Legacy ChArUco pattern (leave unchecked for the default nonlegacy board).</span></label>
        <div class="tool-actions"><button id="calibration-defaults" type="button" class="ghost-button">Reset board defaults</button></div>
        <label class="tool-checkbox"><input id="board-confirmed" type="checkbox" /><span>I checked the printed square and marker sizes, dictionary, IDs, and pattern against the board settings above.</span></label>
        </details>
        <details id="calibration-noise-step" class="tool-details"><summary>Stationary recordings</summary>
          <div class="tool-actions"><button id="noise-start" type="button" class="secondary-button hidden" disabled>Record stationary sensors (~60s)</button><button id="noise-stop" type="button" class="danger-button hidden" disabled>Stop and retain partial</button><button id="noise-upload" type="button" class="ghost-button hidden" disabled>Upload selected recording</button></div>
          <p id="noise-native-status" class="tool-note" role="status">Use the Android app for native stationary acquisition, then upload the retained sensor-only ZIP from Library.</p>
          <div class="tool-grid"><label class="tool-field tool-wide"><span>Retained IMU recording on the server</span><select id="noise-recording" class="text-input"><option value="">Refresh retained recordings…</option></select></label><label class="tool-field tool-wide"><span>Use a completed noise job (optional)</span><select id="calibration-noise-job" class="text-input"><option value="">Choose a completed noise job…</option></select></label><details class="tool-details tool-wide"><summary>Enter a noise-calibration ID manually</summary><label class="tool-field"><span>Noise-calibration job ID</span><input id="calibration-noise-id" class="text-input" maxlength="160" autocomplete="off" /></label></details></div>
          <div class="tool-actions"><button id="noise-refresh" type="button" class="ghost-button">Refresh IMU recordings</button><button id="noise-process" type="button" class="secondary-button" disabled>Process stationary noise</button></div>
          <p id="noise-server-status" class="tool-note" role="status">Short or incomplete recordings remain evidence. A finished job does not prove noise qualification or metric camera–IMU calibration.</p>
        </details>
        <p class="tool-note">RGB room walks remain available without calibration. Metric VIO additionally needs a verified matching full profile and first-walk validation. No action here publishes a live Noesis world.</p>
      </form></details>
      <details id="calibration-profile-details" class="tool-details"><summary>Check and reuse a motion profile</summary>
      <section id="calibration-vio-profile" class="camera-selection-card" tabindex="-1"><h3>Reusable short-session motion profile</h3>
        <p class="tool-note">After processing all three takes, choose their job references and check the fixed profile with OpenVINS on held-out board motion. Long-term noise measurement is not a prerequisite. This check is not a five-minute room-accuracy certificate.</p>
        <p id="motion-profile-capability" class="tool-note" role="status">Profile-check capability is not reported yet. The server checks backend availability when you submit.</p>
        <p id="motion-profile-missing" class="tool-note hidden"></p><div class="tool-actions"><button id="motion-continue-missing" type="button" class="primary-button hidden">Continue setup</button><button id="motion-results-refresh" type="button" class="ghost-button">Refresh calibration results</button></div>
        <form id="motion-profile-form"><div class="tool-grid">
          <label class="tool-field"><span>Camera job</span><select id="motion-camera-job" class="text-input"><option value="">Choose a completed camera job…</option></select></label>
          <label class="tool-field"><span>Camera–IMU job</span><select id="motion-imu-job" class="text-input"><option value="">Choose a completed camera–IMU job…</option></select></label>
          <label class="tool-field tool-wide"><span>Noise job · short-session candidate is sufficient</span><select id="motion-noise-job" class="text-input"><option value="">Choose a completed noise job…</option></select></label>
        </div><div class="tool-actions"><button id="motion-profile-check" class="primary-button" type="submit" disabled>Check motion profile</button></div></form>
        <p id="calibration-vio-profile-status" class="tool-note" role="status">Checking the selected motion profile…</p>
        <div class="tool-actions"><button id="motion-selection-refresh" type="button" class="ghost-button">Refresh motion selection</button><button id="motion-selection-clear" type="button" class="ghost-button hidden" disabled>Clear motion selection</button></div>
        <p class="tool-note">Keep the same phone, physical lens, resolution, orientation, zoom/crop and locked focus. Selection applies to new matching native imports only; no old scans are rewritten. Review the first normal 1–5 minute walk separately before relying on its metric result.</p>
      </section></details>
    <details id="calibration-history" class="tool-details"><summary>Saved jobs and profile settings</summary><section class="calibration-jobs"><div class="tool-heading"><h3>Calibration jobs</h3><button id="calibration-refresh" class="ghost-button" type="button">Refresh jobs</button></div><section class="camera-selection-card"><strong>Camera profile for future matching walks</strong><p id="calibration-camera-selection" class="tool-note" role="status">Checking the server selection…</p><button id="calibration-clear-camera-selection" type="button" class="ghost-button hidden">Clear selection</button></section><p id="calibration-job-status" class="tool-note" role="status">Open Setup to check server availability.</p><div id="calibration-jobs" class="calibration-job-list"></div></section></details>
      <details class="tool-details"><summary>About these checks</summary><p>Recording and processing are separate. A finished take is not a calibration pass. Short-session noise does not measure long-term drift; a passing motion profile still needs a matching normal walk. These steps never change live Noesis tracking calibration.</p></details>
      </div></details>`;
  }

  boardValues() {
    const value = (selector) => this.element(selector).value;
    return {
      squares_x: value("#board-squares-x"), squares_y: value("#board-squares-y"),
      square_length_m: value("#board-square-length"), marker_length_m: value("#board-marker-length"),
      dictionary: value("#board-dictionary"), marker_ids: value("#board-marker-ids"),
      legacy_pattern: this.element("#board-legacy").checked,
    };
  }

  fillBoard(board) {
    const fields = { squares_x: "#board-squares-x", squares_y: "#board-squares-y", square_length_m: "#board-square-length", marker_length_m: "#board-marker-length", dictionary: "#board-dictionary", marker_ids: "#board-marker-ids" };
    for (const [key, selector] of Object.entries(fields)) {
      this.element(selector).value = board[key] === undefined ? "" : Array.isArray(board[key]) ? board[key].join(", ") : String(board[key]);
    }
    this.element("#board-legacy").checked = board.legacy_pattern === true;
    this.element("#board-confirmed").checked = false;
    this.updateBoardSummary();
  }

  updateBoardSummary() {
    const summary = this.element("#calibration-board-summary");
    if (!summary) return;
    try {
      const board = validateBoard(this.boardValues()), ids = board.marker_ids;
      const contiguous = ids.every((id, index) => id === ids[0] + index);
      const label = contiguous ? `${ids[0]}–${ids.at(-1)}` : `${ids.slice(0, 4).join(", ")}${ids.length > 4 ? "…" : ""}`;
      summary.textContent = `${board.squares_x} × ${board.squares_y} · ${board.square_length_m} m / ${board.marker_length_m} m · IDs ${label} · ${board.legacy_pattern ? "legacy" : "nonlegacy"}`;
    } catch (_) { summary.textContent = "Board settings need review · expand to edit"; }
  }

  restoreForm() {
    try {
      const saved = JSON.parse(this.platform.localStorage?.getItem(FORM_KEY) || "null");
      if (saved?.board) this.fillBoard(validateBoard(saved.board));
      this.element("#board-confirmed").checked = saved?.board_geometry_confirmed === true;
      if (["camera", "imu"].includes(saved?.mode)) this.element("#calibration-mode").value = saved.mode;
      if (typeof saved?.camera_calibration_id === "string") this.element("#calibration-camera-id").value = saved.camera_calibration_id.slice(0, 160);
      if (typeof saved?.noise_calibration_id === "string") this.element("#calibration-noise-id").value = saved.noise_calibration_id.slice(0, 160);
      for (const mode of ["camera", "imu", "noise"]) if (typeof saved?.motion_sources?.[mode] === "string") this.motionSourceChoices[mode] = saved.motion_sources[mode].slice(0, 160);
    } catch (_) { /* Invalid or unavailable local storage leaves explicit defaults. */ }
  }

  saveForm() {
    this.updateBoardSummary();
    if (this.jobsLoaded) for (const mode of ["camera", "imu", "noise"]) this.motionSourceChoices[mode] = this.element(`#motion-${mode}-job`).value;
    try { this.platform.localStorage?.setItem(FORM_KEY, JSON.stringify({ board: validateBoard(this.boardValues()), board_geometry_confirmed: this.element("#board-confirmed").checked, mode: this.element("#calibration-mode").value, camera_calibration_id: this.element("#calibration-camera-id").value.trim(), noise_calibration_id: this.element("#calibration-noise-id").value.trim(), motion_sources: this.motionSourceChoices })); }
    catch (_) { /* An incomplete form is editable; it is never sent implicitly. */ }
  }

  message(text, error = false) {
    const element = this.element("#calibration-message");
    if (element) { element.textContent = text; element.classList.toggle("hidden", !text); element.classList.toggle("error-box", error); }
  }

  setCapabilities(capabilities, { offline = false } = {}) {
    const status = this.element("#calibration-capabilities");
    if (!status) return;
    status.classList.remove("hidden");
    const motionCapability = this.element("#motion-profile-capability");
    if (offline) { status.textContent = "Server unavailable. Local native capture remains available; processing needs a server connection."; motionCapability.textContent = "Reconnect to the Noesis server to check its OpenVINS backend and process the retained takes. Local recording remains available."; return; }
    if (capabilities?.schema !== "roomwalk.calibration_capabilities.v1") { status.textContent = "Server capability details are not reported. Local native capture remains available."; return; }
    const state = (value) => value === true ? "available" : value === false ? "not configured" : "not reported";
    status.textContent = `Server: camera processing ${state(capabilities.camera_processing)} · stationary noise ${state(capabilities.stationary_noise_processing)} · camera–IMU solver ${state(capabilities.camera_imu_solver_configured)}.${capabilities.automatic_metric_vio_admission === false ? " No automatic metric-VIO admission." : ""}`;
    status.classList.toggle("hidden", capabilities.camera_processing === true && capabilities.stationary_noise_processing === true && capabilities.camera_imu_solver_configured === true);
    this.motionProfileAvailable = typeof capabilities.motion_profile_validation_available === "boolean" ? capabilities.motion_profile_validation_available : undefined;
    motionCapability.textContent = this.motionProfileAvailable === false
      ? `Motion-profile checking is unavailable. ${typeof capabilities.motion_profile_unavailable_reason === "string" ? capabilities.motion_profile_unavailable_reason.slice(0, 1500) : "The server must have a working OpenVINS validation backend configured."} Keep your recordings and completed jobs; restore the backend, then refresh this page. Recording longer will not fix a missing backend.`
      : this.motionProfileAvailable === true ? "OpenVINS profile checking is available. Choose the three source jobs, then explicitly start the held-out board check."
        : "Profile-check capability is not reported. The server will check OpenVINS availability when you submit; retain your source recordings if it is unavailable.";
    this.updateButtons();
  }

  bind() {
    if (!this.root) return;
    this.root.addEventListener("click", (event) => {
      const phone = event.target.closest?.("[data-recovery-phone]");
      if (phone && this.root.contains(phone) && !phone.disabled) {
        this.pendingResume = { mode: this.recoveryMode, captureId: phone.dataset.recoveryPhone };
        if (!this.nativeCapture?.perform("selectSaved", { id: phone.dataset.recoveryPhone })) this.pendingResume = null;
      }
      const server = event.target.closest?.("[data-recovery-server]");
      if (server && this.root.contains(server)) this.resumeServerTake(server.dataset.recoveryServer);
      const step = event.target.closest?.("[data-calibration-step]");
      if (step && this.root.contains(step)) { this.setGuideStep(step.dataset.calibrationStep); this.element("#calibration-step-picker").open = false; }
    });
    this.element("#calibration-guide-action").addEventListener("click", () => this.openGuideAction());
    this.element("#calibration-guide-next").addEventListener("click", () => {
      const index = CALIBRATION_STEPS.findIndex((step) => step.id === this.guideStep);
      this.setGuideStep(CALIBRATION_STEPS[(index + 1) % CALIBRATION_STEPS.length].id);
    });
    this.element("#motion-profile-form").addEventListener("submit", (event) => { event.preventDefault(); void this.submitMotionProfile(); });
    this.element("#motion-profile-form").addEventListener("change", () => { this.saveForm(); this.updateButtons(); });
    this.element("#motion-results-refresh").addEventListener("click", () => { void this.refreshJobs(); });
    this.element("#motion-continue-missing").addEventListener("click", () => this.openRecovery(this.missingProfileSources()[0]));
    this.element("#calibration-recovery-refresh").addEventListener("click", () => {
      this.nativeCapture?.refresh();
      void this.refreshJobs(); void this.refreshImuRecordings();
      void this.request("/api/scans").then((scans) => this.setScans(scans)).catch((error) => this.message(`Could not refresh uploaded recordings: ${error.message}`, true));
    });
    this.element("#calibration-recovery-record").addEventListener("click", () => { this.recoveryMode = null; this.workflow = null; this.renderGuide(); this.openGuideAction(); });
    this.element("#motion-selection-refresh").addEventListener("click", () => { void this.refreshMotionSelection(); });
    this.element("#motion-selection-clear").addEventListener("click", () => { void this.selectMotionProfile(null); });
    this.element("#calibration-form").addEventListener("input", (event) => {
      this.captureRequestError = "";
      if (["board-squares-x", "board-squares-y", "board-square-length", "board-marker-length", "board-dictionary", "board-marker-ids", "board-legacy"].includes(event.target.id)) this.element("#board-confirmed").checked = false;
      this.saveForm();
      this.updateButtons();
    });
    this.element("#calibration-mode").addEventListener("change", () => { this.saveForm(); this.updateNative(); if (this.element("#calibration-mode").value === "imu") void this.refreshImuRecordings(); });
    this.element("#calibration-scan").addEventListener("change", () => { this.element("#board-confirmed").checked = false; this.loadSelectedScanRequest(); this.updateButtons(); });
    this.element("#calibration-camera-job").addEventListener("change", (event) => {
      this.element("#calibration-camera-id").value = event.currentTarget.value;
      this.saveForm();
      this.updateButtons();
    });
    this.element("#calibration-noise-job").addEventListener("change", (event) => { this.element("#calibration-noise-id").value = event.currentTarget.value; this.saveForm(); });
    this.element("#noise-recording").addEventListener("change", () => this.updateButtons());
    this.element("#noise-start").addEventListener("click", () => {
      if (this.nativeCapture?.startImu(1)) { this.message("Starting shortly. Set the phone down and leave it untouched."); this.focusProgress(); }
    });
    this.element("#noise-stop").addEventListener("click", () => this.nativeCapture?.perform("stopImu"));
    this.element("#noise-stop-top").addEventListener("click", () => this.nativeCapture?.perform("stopImu"));
    this.element("#noise-upload").addEventListener("click", () => this.nativeCapture?.perform("upload"));
    this.element("#noise-refresh").addEventListener("click", () => { void this.refreshImuRecordings(); });
    this.element("#noise-process").addEventListener("click", () => { void this.submitNoise(); });
    this.element("#calibration-form").addEventListener("submit", (event) => { event.preventDefault(); void this.submit(); });
    this.element("#calibration-defaults").addEventListener("click", () => {
      this.captureRequestError = "";
      this.fillBoard(DEFAULT_BOARD);
      this.element("#board-marker-ids").value = "300..369";
      this.saveForm();
      this.message("Board defaults restored. Check the actual printed target before confirming its geometry.");
      this.updateButtons();
    });
    const openNativeCalibration = () => {
      try {
        if (this.captureRequestError) throw new Error(this.captureRequestError);
        const mode = this.element("#calibration-mode").value;
        const missing = this.motionPrerequisite(mode, true);
        if (missing) { this.reviewMotionPrerequisite(); throw new Error(missing); }
        const board = validateBoard(this.boardValues());
        this.saveForm();
        const options = { board_geometry_confirmed: this.element("#board-confirmed").checked, camera_calibration_id: this.element("#calibration-camera-id").value.trim(), noise_calibration_id: this.element("#calibration-noise-id").value.trim() };
        if (this.nativeCapture?.capture(mode, board, options)) this.message(`Opening ${mode === "imu" ? "60–90-second dynamic phone movement" : "~60-second camera coverage"}. Leave the BOARD FIXED; move the PHONE. ${mode === "imu" ? "Do not unlock or relock focus: it must match the selected camera calibration." : "Lock focus before Record."} The 10-second test remains available. Keep this lens and locked focus for later walks. Saved captures open in Library.`);
      } catch (error) { this.message(error.message, true); }
    };
    this.element("#calibration-capture").addEventListener("click", openNativeCalibration);
    this.element("#calibration-test").addEventListener("click", openNativeCalibration);
    this.element("#calibration-refresh").addEventListener("click", () => { void this.refreshJobs(); void this.refreshCameraSelection(); void this.refreshMotionSelection(); });
    this.element("#calibration-clear-camera-selection").addEventListener("click", () => { void this.selectCameraProfile(null); });
    this.element("#calibration-jobs").addEventListener("click", (event) => {
      const button = event.target.closest?.("[data-calibration-job]");
      if (button && this.root.contains(button)) {
        const job = this.jobs.find((job) => jobId(job) === button.dataset.calibrationJob);
        this.setGuideStep(["noise", "camera", "imu"].includes(job?.mode) ? job.mode : "profile", { focus: false });
        void this.loadJob(button.dataset.calibrationJob);
      }
    });
    this.element("#calibration-job-detail").addEventListener("click", (event) => {
      const motion = event.target.closest?.("[data-motion-select]");
      if (motion && this.root.contains(motion) && !motion.disabled) void this.selectMotionProfile(motion.dataset.motionSelect);
      const profile = event.target.closest?.("[data-motion-source]");
      if (profile && this.root.contains(profile)) {
        const job = this.selectedJob;
        if (job?.mode === "imu" && job.status === "completed" && jobId(job) === profile.dataset.motionSource) {
          this.element("#motion-imu-job").value = jobId(job);
          const reference = job.request || job;
          this.element("#motion-camera-job").value = reference.camera_calibration_id || this.element("#calibration-camera-id").value;
          this.element("#motion-noise-job").value = reference.noise_calibration_id || this.element("#calibration-noise-id").value;
          this.setGuideStep("profile");
          this.openGuideAction();
          this.updateButtons();
        }
      }
      const reference = event.target.closest?.("[data-calibration-reference]");
      if (reference && this.root.contains(reference) && !reference.disabled) {
        const job = this.selectedJob;
        if (job?.status === "completed" && jobId(job) === reference.dataset.calibrationReference) {
          if (job.mode === "noise" && (job.result?.noise_model_usable === true || job.result?.imu_noise_calibrated === true)) {
            this.element("#calibration-noise-id").value = jobId(job);
            this.setGuideStep("camera");
          } else if (job.mode === "camera" && job.result?.camera_intrinsics_calibrated === true) {
            this.element("#calibration-camera-id").value = jobId(job);
            this.setGuideStep("imu");
          }
          this.saveForm();
          this.message("Job reference selected for the next stage. The server must still verify the exact phone and capture binding; this is not metric VIO acceptance.");
        }
      }
      const button = event.target.closest?.("[data-calibration-scan]");
      if (button && this.root.contains(button)) this.onOpenScan(button.dataset.calibrationScan);
      const cancel = event.target.closest?.("[data-calibration-cancel]");
      if (cancel && this.root.contains(cancel) && !cancel.disabled) void this.cancelJob(cancel.dataset.calibrationCancel);
      const selection = event.target.closest?.("[data-calibration-select]");
      if (selection && this.root.contains(selection) && !selection.disabled) void this.selectCameraProfile(selection.dataset.calibrationSelect);
    });
  }

  setScans(scans, { preferredId } = {}) {
    this.scans = Array.isArray(scans) ? scans.filter((scan) => typeof scan?.id === "string") : [];
    const select = this.element("#calibration-scan");
    if (!select) return;
    const previous = select.value;
    const choice = preferredId === undefined ? previous : preferredId;
    const selected = this.scans.some((scan) => scan.id === choice) ? choice : "";
    select.innerHTML = '<option value="">Choose an uploaded capture…</option>' + this.scans.map((scan) => `<option value="${escapeHtml(scan.id)}" ${scan.id === selected ? "selected" : ""}>${escapeHtml(scan.name || scan.id)} · ${escapeHtml(scan.status || "unknown")}</option>`).join("");
    select.value = selected;
    if (previous !== selected) this.element("#board-confirmed").checked = false;
    this.loadSelectedScanRequest();
    this.updateButtons();
  }

  loadSelectedScanRequest() {
    const id = this.element("#calibration-scan")?.value;
    const request = this.scans.find((scan) => scan.id === id)?.capture?.calibration_request;
    const key = JSON.stringify([id, request]);
    if (key === this.scanRequestKey) return;
    this.scanRequestKey = key;
    this.captureRequestError = "";
    if (request && typeof request === "object") this.applyCaptureRequest(request, `server capture ${id}`, { scanId: id });
  }

  applyCaptureRequest(request, source, { scanId, captureId } = {}) {
    try {
      // Clear absent fields before validating so a malformed sidecar cannot
      // silently inherit the previous take's board. Original evidence is not edited.
      this.fillBoard(request.board && typeof request.board === "object" ? request.board : {});
      this.element("#calibration-mode").value = request.mode;
      if (request.schema !== undefined && request.schema !== CALIBRATION_REQUEST_SCHEMA) throw new Error("The capture uses an unsupported calibration-request schema.");
      const normalized = calibrationRequest({ ...request, scan_id: "selected-capture" });
      this.fillBoard(normalized.board);
      this.captureRequestError = "";
      this.element("#board-confirmed").checked = request.board_geometry_confirmed === true;
      this.element("#calibration-camera-id").value = typeof request.camera_calibration_id === "string" ? request.camera_calibration_id.slice(0, 160) : "";
      this.element("#calibration-noise-id").value = typeof request.noise_calibration_id === "string" ? request.noise_calibration_id.slice(0, 160) : "";
      this.saveForm();
      const flow = this.workflow;
      const matchesWorkflow = flow?.mode === normalized.mode && ((scanId && scanId === flow.scanId) || (captureId && captureId === flow.captureId));
      if (matchesWorkflow) this.guideStep = flow.mode;
      else {
        this.guideStep = "process";
        this.message(`Loaded the saved calibration request for ${source}. Check the selected uploaded capture, then explicitly press Process. No job was started.`);
      }
      this.updateNative();
    } catch (error) {
      this.captureRequestError = `Saved calibration settings need review: ${error.message}`;
      this.element("#calibration-board-settings").open = true;
      this.element("#board-confirmed").checked = false;
      this.element("#calibration-camera-id").value = "";
      this.element("#calibration-noise-id").value = "";
      this.message(this.captureRequestError, true);
      this.updateNative();
    }
  }

  updateButtons() {
    const process = this.element("#calibration-process");
    if (process) process.disabled = this.submitting || Boolean(this.captureRequestError) || !this.scans.some((scan) => scan.id === this.element("#calibration-scan").value);
    for (const id of ["#calibration-capture", "#calibration-test"]) {
      const capture = this.element(id);
      if (capture) capture.disabled = !this.nativeCapture?.enabled("capture") || this.nativeCapture.configDirty || Boolean(this.captureRequestError);
    }
    const noise = this.element("#noise-process");
    if (noise) noise.disabled = this.noiseSubmitting || !this.imuRecordings.some((receipt) => imuCaptureId(receipt) === this.element("#noise-recording").value);
    if (noise) noise.textContent = this.noiseSubmitting ? "Submitting…" : "Process stationary noise";
    const motion = this.element("#motion-profile-check");
    if (motion) motion.disabled = !this.jobsLoaded || this.loading || this.motionSubmitting || this.motionProfileAvailable === false || ["camera", "imu", "noise"].some((mode) => !this.jobs.some((job) => job.mode === mode && job.status === "completed" && jobId(job) === this.element(`#motion-${mode}-job`).value));
    this.renderGuide();
  }

  updateNative(snapshot) {
    const available = this.nativeCapture?.available === true;
    this.element("#calibration-capture")?.classList.toggle("hidden", !available);
    this.element("#calibration-test")?.classList.toggle("hidden", !available);
    this.element("#calibration-camera-job")?.closest("label").classList.toggle("hidden", this.element("#calibration-mode")?.value !== "imu");
    this.element("#calibration-camera-advanced")?.classList.toggle("hidden", this.element("#calibration-mode")?.value !== "imu");
    for (const [id, action] of [["#noise-start", "startImu"], ["#noise-stop", "stopImu"], ["#noise-upload", "upload"], ["#noise-stop-top", "stopImu"]]) {
      const button = this.element(id);
      button?.classList.toggle("hidden", !available);
      if (button) button.disabled = !this.nativeCapture?.enabled(action);
    }
    const native = snapshot || this.nativeCapture?.snapshot;
    const resume = this.pendingResume;
    if (snapshot && resume && !native.busy && !native.active && !native.imu_active && native.selected_capture === resume.captureId) {
      this.pendingResume = null;
      const request = selectedCalibrationRequest(native);
      const source = native.saved?.find((take) => take.id === resume.captureId);
      const matches = resume.mode === "noise" ? source?.kind === "imu_diagnostic" : request?.mode === resume.mode && request.capture_id === resume.captureId && request.short_test !== true;
      if (matches) {
        this.recoveryMode = null;
        this.workflow = { ...resume, phase: native.artifact?.name?.endsWith(".zip") ? "saved" : "needs_package" };
        if (resume.mode !== "noise" && native.capture_outcome?.capture_id === resume.captureId && native.capture_outcome.partial === true) this.workflow.phase = "capture_interrupted";
        this.guideStep = resume.mode;
        this.message(this.workflow.phase === "capture_interrupted" ? native.capture_outcome.message : "", this.workflow.phase === "capture_interrupted");
        this.openSection("#calibration-guide-content");
      } else this.message("The selected take's calibration settings could not be restored. Keep the original and review its phone status; no upload was started.", true);
    }
    const sensorRun = Boolean(native?.imu_active || native?.imu_preparing);
    this.element("#noise-stop-top")?.classList.toggle("hidden", !sensorRun);
    if (this.element("#noise-stop-top")) this.element("#noise-stop-top").textContent = native?.imu_preparing ? "Cancel start" : "Stop and retain partial";
    const progress = this.element("#calibration-guide-progress");
    if (progress) progress.innerHTML = captureGuidanceMarkup(native || {});
    this.element("#calibration-run")?.classList.toggle("hidden", !(sensorRun || native?.active));
    if (!sensorRun && this.sensorRunActive && /cancelled/.test(native?.status || "")) this.message("Start cancelled. No new recording was made.");
    if (sensorRun && !this.sensorRunActive && this.visible) this.focusProgress();
    this.sensorRunActive = sensorRun;
    const flow = this.workflow, transfer = native?.transfer;
    if (snapshot && flow?.phase === "packaging" && native.selected_capture === flow.captureId && !native.busy) {
      flow.phase = native.artifact?.name?.endsWith(".zip") ? "saved" : "needs_package";
      if (flow.phase === "needs_package") this.message(native.detail || "Packaging did not finish. The raw recording is retained.", true);
    }
    if (flow && native?.selected_capture === flow.captureId && transfer?.archive_name === native.artifact?.name) {
      const receiptCaptureId = transfer.receipt?.capture_id || transfer.receipt?.upload_receipt?.capture_id;
      if (transfer.state === "complete" && receiptCaptureId === flow.captureId && ["saved", "uploading"].includes(flow.phase)) {
        flow.phase = "uploaded"; flow.scanId = transfer.receipt.id;
        this.message("Upload complete. Process this recording to continue.");
      } else if (["failed", "cancelled", "interrupted"].includes(transfer.state) && flow.phase === "uploading") {
        flow.phase = "saved"; this.message(transfer.error || "Upload stopped. Your recording is still saved; try uploading again.", true);
      } else if (transfer.active && flow.phase === "uploading") {
        const total = Number(transfer.total_bytes), sent = Number(transfer.sent_bytes);
        this.message(total > 0 && sent >= 0 ? `Uploading · ${Math.min(100, Math.floor(sent / total * 100))}%` : "Uploading…");
      }
    }
    if (available && native) this.element("#noise-native-status").textContent = `${native.status || "Native recorder"}${native.imu_progress ? ` · ${typeof native.imu_progress === "string" ? native.imu_progress : JSON.stringify(native.imu_progress)}` : ""}${native.artifact?.name ? ` · Selected artifact: ${native.artifact.name}. Verify this is the sensor-only recording before uploading.` : ""}`;
    const request = selectedCalibrationRequest(snapshot);
    if (snapshot) {
      this.hasNativeCalibrationRequest = Boolean(request);
      const captureId = typeof snapshot.selected_capture === "string" ? snapshot.selected_capture : snapshot.selected_capture?.id || "";
      const key = JSON.stringify([captureId, request]);
      if (request && key !== this.nativeRequestKey) {
        this.nativeRequestKey = key;
        this.applyCaptureRequest(request, captureId ? `phone take ${captureId}` : "the selected phone take", { captureId });
      } else if (!request) this.nativeRequestKey = "";
      const outcome = snapshot.capture_outcome;
      if (!snapshot.active && !snapshot.busy && outcome?.capture_id === captureId && outcome.partial === true && ["camera", "imu"].includes(request?.mode)) {
        const outcomeKey = JSON.stringify([captureId, outcome.stop_reason]);
        if (outcomeKey !== this.nativeOutcomeKey) {
          this.nativeOutcomeKey = outcomeKey;
          this.recoveryMode = null;
          this.workflow = { mode: request.mode, captureId, phase: "capture_interrupted" };
          this.guideStep = request.mode;
          this.message(outcome.message || "Recording stopped early. The original is retained in Library.", true);
        }
      }
    }
    if (available) {
      const focusLocked = native?.cameras?.[native.camera_index]?.focus_locked === true;
      this.element("#calibration-capture").textContent = focusLocked ? "Open calibration camera" : "Open camera to lock focus";
      this.element("#calibration-capture-note").textContent = focusLocked
        ? "Focus is locked. Keep the BOARD FIXED and move the PHONE. Keep this lens and locked focus for both camera stages and later walks. The native viewer provides the 10-second test and guided Record controls."
        : "Keep the BOARD FIXED; move the PHONE for both camera stages. In the native viewer, lock focus before Record. The 10-second test remains available. Keep this lens and locked focus for later walks.";
    }
    this.updateButtons();
  }

  setVisible(visible) {
    this.visible = visible;
    clearTimeout(this.pollTimer);
    if (visible) { void this.refreshJobs(); void this.refreshCameraSelection(); void this.refreshMotionSelection(); void this.refreshImuRecordings(); }
  }

  request(url, options) { return calibrationJsonFetch(url, options, { fetchImpl: this.fetchImpl }); }

  setMotionSelection(payload) {
    const selection = payload?.selection;
    if (!payload || !Object.hasOwn(payload, "selection") || (selection !== null && (typeof selection?.motion_calibration_id !== "string" || !selection.motion_calibration_id || typeof selection.profile_id !== "string" || !selection.profile_id || selection.status !== "ready_for_short_walks" || !Number.isFinite(selection.maximum_duration_s) || selection.maximum_duration_s <= 0 || selection.maximum_duration_s > 300))) throw new Error("The server returned an unsupported motion-profile selection. Refresh before using it.");
    this.motionSelection = selection;
    this.motionSelectionLoaded = true;
    this.renderMotionSelection();
  }

  renderMotionSelection() {
    const selection = this.motionSelection;
    this.element("#calibration-vio-profile-status").textContent = !this.motionSelectionLoaded
      ? "Motion selection is not confirmed. Refresh motion selection to read the server state before relying on it."
      : selection
      ? `Selected motion profile: ${selection.motion_calibration_id.slice(0, 160)} · ${selection.profile_id.slice(0, 160)}. Ready for matching short walks up to ${readableDuration(selection.maximum_duration_s)}. Only new matching native imports use it; old scans are unchanged. Review the first normal walk separately.`
      : "No motion profile selected. Run Check motion profile, then explicitly use a completed ready result for matching room walks. Camera-only selection does not complete this step.";
    const clear = this.element("#motion-selection-clear");
    clear.classList.toggle("hidden", !selection);
    clear.disabled = this.motionSelecting || !this.motionSelectionLoaded;
  }

  async refreshMotionSelection() {
    if (this.motionSelecting || this.motionSelectionLoading) return;
    this.motionSelectionLoading = true;
    const revision = this.motionSelectionRevision;
    try {
      const payload = await this.request("/api/calibration/motion-selection");
      if (revision !== this.motionSelectionRevision) return;
      this.setMotionSelection(payload);
      this.renderDetail();
    } catch (error) {
      if (revision !== this.motionSelectionRevision) return;
      this.motionSelectionLoaded = false;
      this.element("#motion-selection-clear").disabled = true;
      this.element("#calibration-vio-profile-status").textContent = `Could not confirm the motion profile: ${error.message} Keep the recordings and restore the server’s motion-profile/OpenVINS support, then refresh. No selection or metric result is confirmed.`;
      this.renderDetail();
    } finally { this.motionSelectionLoading = false; }
  }

  async submitMotionProfile() {
    if (this.motionSubmitting) return;
    try {
      if (this.motionProfileAvailable === false) throw new Error(this.element("#motion-profile-capability").textContent);
      const payload = motionProfileRequest({ camera_calibration_id: this.element("#motion-camera-job").value, imu_calibration_id: this.element("#motion-imu-job").value, noise_calibration_id: this.element("#motion-noise-job").value });
      for (const mode of ["camera", "imu", "noise"]) {
        if (!this.jobs.some((job) => job.mode === mode && job.status === "completed" && jobId(job) === payload[`${mode}_calibration_id`])) throw new Error("Choose currently listed completed source jobs. The server must verify their qualification and binding.");
      }
      this.motionSubmitting = true;
      this.updateButtons();
      const response = await this.request("/api/calibration/motion-profile-jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const job = response?.job || response;
      if (!jobId(job) || job.mode !== "motion_profile") throw new Error("The server did not confirm a motion-profile job. Refresh jobs before retrying.");
      this.selectedJobId = jobId(job);
      this.selectedJob = job;
      this.renderDetail();
      this.message("Motion-profile check queued. OpenVINS will test the fixed profile on held-out board motion. Wait for a completed ready result before selecting it; this is not a five-minute accuracy certificate.");
      await this.refreshJobs();
    } catch (error) { this.message(`Motion-profile check was not confirmed: ${error.message} Your source recordings and jobs are retained.`, true); }
    finally { this.motionSubmitting = false; this.updateButtons(); }
  }

  async selectMotionProfile(id) {
    if (this.motionSelecting) return;
    const job = this.selectedJob;
    if (id !== null && (jobId(job) !== id || job?.mode !== "motion_profile" || job.status !== "completed" || job.result?.motion_profile_ready !== true)) return;
    this.motionSelecting = true;
    this.motionSelectionRevision += 1;
    this.renderMotionSelection();
    this.renderDetail();
    try {
      const payload = await this.request("/api/calibration/motion-selection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ motion_calibration_id: id }) });
      if (id === null ? payload?.selection !== null : payload?.selection?.motion_calibration_id !== id) throw new Error("The server did not confirm the requested motion selection.");
      this.setMotionSelection(payload);
      if (id !== null) this.setGuideStep("walk");
      this.message(id === null ? "Future-import motion selection cleared. Old scans remain unchanged." : "Motion profile selected for matching new room-walk imports. Keep the same phone configuration and review the first normal walk separately; old scans and live Noesis settings are unchanged.");
    } catch (error) {
      this.motionSelectionLoaded = false;
      this.element("#calibration-vio-profile-status").textContent = "Motion selection is unconfirmed. Refresh motion selection to read the server state before relying on it.";
      this.message(`Motion selection was not confirmed: ${error.message} Refresh motion selection before trying again.`, true);
    } finally {
      this.motionSelecting = false;
      if (this.motionSelectionLoaded) this.renderMotionSelection();
      this.element("#motion-selection-clear").disabled = !this.motionSelectionLoaded;
      this.renderDetail();
    }
  }

  setCameraSelection(payload) {
    if (!payload || !Object.hasOwn(payload, "selection") || (payload.selection !== null && (typeof payload.selection?.camera_calibration_id !== "string" || typeof payload.selection?.profile_id !== "string"))) throw new Error("The server returned an unsupported camera-profile selection.");
    this.cameraSelection = payload.selection;
    this.cameraSelectionLoaded = true;
    this.renderCameraSelection();
  }

  renderCameraSelection() {
    const selection = this.cameraSelection;
    this.element("#calibration-camera-selection").textContent = selection
      ? `Selected: ${selection.camera_calibration_id.slice(0, 160)} · ${selection.profile_id.slice(0, 160)}. Only new native imports with matching device, lens, locked focus, and crop can use this profile. Existing scans and live Noesis settings are unchanged.`
      : "No camera profile selected for future imports. Selecting a qualified camera job does not promote live calibration or enable metric VIO.";
    const clear = this.element("#calibration-clear-camera-selection");
    clear.classList.toggle("hidden", !selection);
    clear.disabled = this.cameraSelecting;
  }

  async refreshCameraSelection() {
    if (this.cameraSelecting || this.cameraSelectionLoading) return;
    this.cameraSelectionLoading = true;
    const revision = this.cameraSelectionRevision;
    try { const payload = await this.request("/api/calibration/camera-selection"); if (revision !== this.cameraSelectionRevision) return; this.setCameraSelection(payload); this.renderDetail(); }
    catch (error) { if (revision !== this.cameraSelectionRevision) return; this.cameraSelectionLoaded = false; this.element("#calibration-clear-camera-selection").disabled = true; this.element("#calibration-camera-selection").textContent = `Could not confirm the current camera profile: ${error.message}`; this.renderDetail(); }
    finally { this.cameraSelectionLoading = false; }
  }

  async selectCameraProfile(id) {
    if (this.cameraSelecting) return;
    const job = this.selectedJob;
    if (id !== null && (jobId(job) !== id || job.mode !== "camera" || job.result?.camera_intrinsics_calibrated !== true || !safeReportUrl(job.artifacts?.camera_profile_url, this.platform.location?.origin))) return;
    this.cameraSelecting = true;
    this.cameraSelectionRevision += 1;
    this.renderCameraSelection();
    this.renderDetail();
    try {
      const payload = await this.request("/api/calibration/camera-selection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ camera_calibration_id: id }) });
      if (id === null ? payload?.selection !== null : payload?.selection?.camera_calibration_id !== id) throw new Error("The server did not confirm the requested camera selection.");
      this.setCameraSelection(payload);
      this.message(id === null ? "Future-import camera selection cleared. Existing scans are unchanged." : "Camera profile selected for matching new native imports only. The server checks the exact capture binding; existing scans, live calibration, and metric VIO are unchanged.");
    } catch (error) {
      this.cameraSelectionLoaded = false;
      this.message(`Camera selection was not confirmed: ${error.message} Refresh jobs to check the server before trying again.`, true);
      this.element("#calibration-camera-selection").textContent = "Camera selection is unconfirmed. Refresh jobs to read the server's selection.";
    } finally {
      this.cameraSelecting = false;
      if (this.cameraSelectionLoaded) this.renderCameraSelection();
      this.element("#calibration-clear-camera-selection").disabled = !this.cameraSelectionLoaded;
      this.renderDetail();
    }
  }

  async submit() {
    if (this.submitting) return;
    try {
      if (this.captureRequestError) throw new Error(this.captureRequestError);
      const scan_id = this.element("#calibration-scan").value;
      if (!this.scans.some((scan) => scan.id === scan_id)) throw new Error("Choose a currently listed uploaded capture.");
      const payload = calibrationRequest({ scan_id, mode: this.element("#calibration-mode").value, board: this.boardValues(), board_geometry_confirmed: this.element("#board-confirmed").checked, camera_calibration_id: this.element("#calibration-camera-id").value.trim(), noise_calibration_id: this.element("#calibration-noise-id").value.trim() });
      const missing = this.motionPrerequisite(payload.mode);
      if (missing) { this.reviewMotionPrerequisite(); throw new Error(missing); }
      this.submitting = true;
      this.updateButtons();
      this.saveForm();
      const response = await this.request("/api/calibration/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const job = response?.job || response;
      if (jobId(job)) { this.selectedJobId = jobId(job); this.selectedJob = job; if (this.workflow?.scanId === scan_id) { this.workflow.jobId = jobId(job); this.workflow.phase = "processing"; } this.setGuideStep(job.mode, { focus: false }); this.renderDetail(); }
      this.message("Processing started. The result will appear here.");
      if (!this.sensorRunActive) this.openSection("#calibration-guide-content");
      await this.refreshJobs();
    } catch (error) { this.message(error.message || "Calibration processing is unavailable. Your capture remains retained.", true); }
    finally { this.submitting = false; this.updateButtons(); }
  }

  async refreshImuRecordings() {
    if (this.noiseLoading) return;
    this.noiseLoading = true;
    this.element("#noise-refresh").disabled = true;
    try {
      const payload = await this.request("/api/calibration/imu-recordings");
      if (!Array.isArray(payload?.recordings)) throw new Error("The server returned an unsupported IMU recording list.");
      this.imuRecordings = payload.recordings.filter((receipt) => typeof imuCaptureId(receipt) === "string" && imuCaptureId(receipt));
      const select = this.element("#noise-recording");
      const previous = select.value;
      select.innerHTML = '<option value="">Choose a retained sensor recording…</option>' + this.imuRecordings.map((receipt) => `<option value="${escapeHtml(imuCaptureId(receipt))}" ${imuCaptureId(receipt) === previous ? "selected" : ""}>${escapeHtml(imuCaptureId(receipt))}</option>`).join("");
      this.element("#noise-server-status").textContent = `${this.imuRecordings.length} retained IMU recording${this.imuRecordings.length === 1 ? "" : "s"}. Select the intended sensor capture and explicitly process it; the server checks duration, stillness, and qualification evidence.`;
    } catch (error) {
      this.element("#noise-server-status").textContent = `Could not refresh stationary recordings: ${error.message}`;
    } finally { this.noiseLoading = false; this.element("#noise-refresh").disabled = false; this.updateButtons(); }
  }

  async submitNoise() {
    if (this.noiseSubmitting) return;
    try {
      const id = this.element("#noise-recording").value;
      if (!this.imuRecordings.some((receipt) => imuCaptureId(receipt) === id)) throw new Error("Choose a currently listed retained IMU recording.");
      this.noiseSubmitting = true;
      this.message("Submitting stationary recording…");
      this.updateButtons();
      const response = await this.request("/api/calibration/noise-jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ imu_capture_id: id }) });
      const job = response?.job || response;
      if (jobId(job)) { this.selectedJobId = jobId(job); this.selectedJob = job; if (this.workflow?.captureId === id) { this.workflow.jobId = jobId(job); this.workflow.phase = "processing"; } this.setGuideStep("noise", { focus: false }); this.renderDetail(); }
      this.message("Processing started. The result will appear here.");
      if (!this.sensorRunActive) this.openSection("#calibration-guide-content");
      await this.refreshJobs();
    } catch (error) { this.message(`Noise processing was not confirmed: ${error.message}`, true); }
    finally { this.noiseSubmitting = false; this.updateButtons(); }
  }

  async refreshJobs() {
    if (this.loading) return;
    this.loading = true;
    this.jobsError = "";
    this.updateButtons();
    clearTimeout(this.pollTimer);
    const refresh = this.element("#calibration-refresh");
    if (refresh) refresh.disabled = true;
    let succeeded = false;
    try {
      const payload = await this.request("/api/calibration/jobs");
      if (!Array.isArray(payload?.jobs)) throw new Error("The calibration server returned an unsupported job list.");
      this.jobs = payload.jobs.filter((job) => job && typeof job === "object");
      for (const mode of ["camera", "imu", "noise"]) {
        const select = this.element(`#motion-${mode}-job`), previous = select.value || this.motionSourceChoices[mode] || (mode === "imu" ? "" : this.element(`#calibration-${mode === "noise" ? "noise" : "camera"}-id`).value);
        select.innerHTML = `<option value="">Choose a completed ${mode === "imu" ? "camera–IMU" : mode} job…</option>` + this.jobs.filter((job) => job.mode === mode && job.status === "completed" && jobId(job)).map((job) => `<option value="${escapeHtml(jobId(job))}" ${jobId(job) === previous ? "selected" : ""}>${escapeHtml(jobId(job))}${mode === "noise" ? job.result?.noise_model_usable === true ? " · usable short-session candidate" : " · review suitability" : ""}</option>`).join("");
      }
      const cameraJobs = this.element("#calibration-camera-job");
      const selectedCameraJob = this.element("#calibration-camera-id").value;
      cameraJobs.innerHTML = '<option value="">Choose a qualified camera job or enter its ID below…</option>' + this.jobs.filter((job) => job.mode === "camera" && job.status === "completed" && job.result?.camera_intrinsics_calibrated === true && jobId(job)).map((job) => `<option value="${escapeHtml(jobId(job))}" ${jobId(job) === selectedCameraJob ? "selected" : ""}>${escapeHtml(jobId(job))} · ${escapeHtml(job.scan_id || "Camera job")}</option>`).join("");
      const selectedNoiseJob = this.element("#calibration-noise-id").value;
      this.element("#calibration-noise-job").innerHTML = '<option value="">Choose a completed noise job…</option>' + this.jobs.filter((job) => job.mode === "noise" && job.status === "completed" && jobId(job)).map((job) => `<option value="${escapeHtml(jobId(job))}" ${jobId(job) === selectedNoiseJob ? "selected" : ""}>${escapeHtml(jobId(job))} · ${job.result?.noise_model_usable === true ? "usable short-session candidate" : job.result?.imu_noise_calibrated === true ? "full noise calibration reported" : "review report before use"}</option>`).join("");
      this.element("#calibration-jobs").innerHTML = calibrationJobCards(this.jobs, this.selectedJobId);
      this.element("#calibration-job-status").textContent = `${this.jobs.length} saved job${this.jobs.length === 1 ? "" : "s"}. Select a job for its status and report. Results remain review-only pending gates.`;
      this.jobsLoaded = true;
      if (this.pollFailures) this.message("Connection restored. Showing the latest server job status.");
      if (this.selectedJobId && !await this.loadJob(this.selectedJobId)) throw new Error("Selected job status could not be refreshed.");
      succeeded = true;
      this.pollFailures = 0;
    } catch (error) {
      this.jobsLoaded = false;
      this.jobsError = `Could not load calibration results: ${error.message || "Server unavailable"}. Refresh results; do not repeat your recordings.`;
      this.element("#calibration-job-status").textContent = error.message || "Server unavailable. Local board editing and native capture remain available.";
      this.pollFailures += 1;
      this.message("Calibration status is temporarily unavailable. Displayed progress may be stale; retrying automatically. Do not restart processing or repeat the recording.", true);
    } finally {
      this.loading = false;
      this.updateButtons();
      if (refresh) refresh.disabled = false;
      if (this.visible && (!succeeded || this.jobs.some((job) => ACTIVE_JOB_STATES.has(job.status)))) this.pollTimer = setTimeout(() => { void this.refreshJobs(); }, succeeded ? 4000 : Math.min(30000, 4000 * 2 ** Math.min(this.pollFailures, 3)));
    }
  }

  async loadJob(id) {
    if (typeof id !== "string" || !id || id.length > 160) return;
    const request = ++this.detailRequest;
    this.selectedJobId = id;
    this.element("#calibration-jobs").innerHTML = calibrationJobCards(this.jobs, id);
    try {
      const payload = await this.request(`/api/calibration/jobs/${encodeURIComponent(id)}`);
      if (request !== this.detailRequest) return;
      const job = payload?.job || payload;
      if (!job || typeof job !== "object" || jobId(job) !== id) throw new Error("Calibration detail does not match the selected job.");
      this.selectedJob = job;
      this.renderDetail();
      return true;
    } catch (error) {
      if (request !== this.detailRequest) return;
      this.selectedJob = null;
      this.element("#calibration-job-detail").textContent = `Could not refresh this job: ${error.message}`;
      return false;
    }
  }

  async cancelJob(id) {
    if (!id || this.cancellingJob || !this.platform.confirm?.("Cancel this calibration job? Original capture evidence will remain retained.")) return;
    this.cancellingJob = id;
    this.renderDetail();
    try {
      await this.request(`/api/calibration/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      this.message("Cancellation requested. The job report will show its final state; capture evidence remains retained.");
      await this.refreshJobs();
    } catch (error) { this.message(`Cancellation could not be confirmed: ${error.message}`, true); }
    finally { this.cancellingJob = null; this.renderDetail(); }
  }

  renderDetail() {
    const job = this.selectedJob;
    const root = this.element("#calibration-job-detail");
    if (!job || !root) return;
    const flow = this.workflow;
    if (flow?.jobId === jobId(job) && !ACTIVE_JOB_STATES.has(job.status)) {
      const report = job.result || {};
      const passed = job.status === "completed" && (job.mode === "noise" ? report.noise_model_usable === true || report.imu_noise_calibrated === true : job.mode === "camera" ? report.camera_intrinsics_calibrated === true : report.camera_imu_extrinsics_calibrated === true && report.time_offset_calibrated === true);
      flow.phase = passed ? "passed" : job.status === "failed" ? "processing_failed" : "retry";
      this.message(passed ? "This take passed its checks. Continue to the next step." : "This take needs attention. See the next action below.", !passed);
      this.renderGuide();
    }
    const rawOpen = root.querySelector("[data-calibration-raw]")?.open === true;
    const openReports = root.dataset.jobId === jobId(job) ? new Set([...root.querySelectorAll("details[open]")].map((item) => item.querySelector("summary")?.textContent)) : new Set();
    const artifacts = { ...(job.artifacts && typeof job.artifacts === "object" ? job.artifacts : {}), ...(job.log_url ? { log_url: job.log_url } : {}), ...(job.report_url ? { report_url: job.report_url } : {}) };
    const links = Object.entries(artifacts).slice(0, 32).map(([key, value]) => {
      const url = safeReportUrl(value, this.platform.location?.origin);
      return url ? `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(humanLabel(key.replace(/_url$/, "")))}</a></li>` : "";
    }).join("");
    const chartUrl = safeReportUrl(artifacts.noise_allan_url, this.platform.location?.origin);
    const chart = chartUrl && /\.(png|svg|webp|jpe?g)$/i.test(new URL(chartUrl).pathname)
      ? `<figure class="result-chart"><a href="${escapeHtml(chartUrl)}" target="_blank" rel="noopener"><img src="${escapeHtml(chartUrl)}" alt="Allan deviation chart comparing training and held-out stationary sensor evidence" loading="lazy" /></a><figcaption>Stationary noise · Allan deviation. Open the chart for full resolution; qualification depends on the reported gates, not the plot alone.</figcaption></figure>` : "";
    const cancel = ["queued", "running", "cancelling"].includes(job.status) ? `<button type="button" class="danger-button" data-calibration-cancel="${escapeHtml(jobId(job))}" ${this.cancellingJob || job.status === "cancelling" ? "disabled" : ""}>${job.status === "cancelling" ? "Cancelling…" : "Cancel job"}</button>` : "";
    const selected = this.cameraSelectionLoaded && this.cameraSelection?.camera_calibration_id === jobId(job);
    const selection = job.mode === "camera" && job.result?.camera_intrinsics_calibrated === true && safeReportUrl(artifacts.camera_profile_url, this.platform.location?.origin)
      ? `<section class="camera-selection-card"><button type="button" class="primary-button" data-calibration-select="${escapeHtml(jobId(job))}" ${selected || this.cameraSelecting ? "disabled" : ""}>${selected ? "Selected for matching new walks" : "Use for future matching walks"}</button><p class="tool-note">Applies only to new native imports whose device, lens, locked focus, and crop match this profile. No existing scans are rewritten; this does not promote live calibration or metric VIO.</p></section>` : "";
    const referenceLabel = job.status === "completed" && job.mode === "noise" && (job.result?.noise_model_usable === true || job.result?.imu_noise_calibrated === true) ? "Use noise reference → camera coverage"
      : job.status === "completed" && job.mode === "camera" && job.result?.camera_intrinsics_calibrated === true ? "Use camera reference → dynamic phone capture" : "";
    const reference = referenceLabel ? `<button type="button" class="secondary-button" data-calibration-reference="${escapeHtml(jobId(job))}">${referenceLabel}</button>` : "";
    const motionSelected = this.motionSelectionLoaded && this.motionSelection?.motion_calibration_id === jobId(job);
    const motionSelection = job.mode === "motion_profile" && job.status === "completed" && job.result?.motion_profile_ready === true
      ? `<section class="camera-selection-card"><button type="button" class="primary-button" data-motion-select="${escapeHtml(jobId(job))}" ${motionSelected || this.motionSelecting ? "disabled" : ""}>${motionSelected ? "Selected for matching room walks" : "Use for matching room walks"}</button><p class="tool-note">Select this checked profile for future matching native room-walk imports only, up to five minutes. The first normal walk still needs review. Existing scans and live Noesis settings are unchanged.</p></section>` : "";
    const motionSource = job.mode === "imu" && job.status === "completed" ? `<button type="button" class="secondary-button" data-motion-source="${escapeHtml(jobId(job))}">Next: check motion profile with this job</button>` : "";
    root.innerHTML = `${calibrationSummaryMarkup(job)}${selection}${motionSelection}<div class="tool-actions">${flow?.jobId === jobId(job) ? "" : reference}${motionSource}${cancel}</div>${chart || links ? `<details class="tool-details"><summary>Charts, source and logs</summary>${chart}${typeof job.scan_id === "string" ? `<button type="button" class="ghost-button" data-calibration-scan="${escapeHtml(job.scan_id)}">Open source capture</button>` : ""}<ul class="result-artifacts">${links}</ul></details>` : ""}<details class="tool-details" data-calibration-raw ${rawOpen ? "open" : ""}><summary>Advanced · raw job status and report</summary><pre class="tool-report"></pre></details>`;
    root.querySelector("pre").textContent = pretty(job);
    root.dataset.jobId = jobId(job);
    for (const report of root.querySelectorAll("details")) if (openReports.has(report.querySelector("summary")?.textContent)) report.open = true;
    root.querySelector(".result-chart img")?.addEventListener("error", () => {
      root.querySelector(".result-chart figcaption").textContent = "The chart preview could not load. Use the noise Allan artifact link to open or save the retained chart.";
    });
    this.renderGuide();
  }
}

export default CalibrationUI;
