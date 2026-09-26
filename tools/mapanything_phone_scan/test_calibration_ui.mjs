import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_BOARD, parseMarkerIds, validateBoard, calibrationRequest,
  calibrationJsonFetch, calibrationJobCards, safeReportUrl, selectedCalibrationRequest,
  CALIBRATION_REQUEST_SCHEMA, calibrationOutcome, calibrationSummaryMarkup,
  scanCameraSelectionMarkup,
  scanMotionProfileMarkup,
  CALIBRATION_STEPS, calibrationNextAction, motionProfileRequest, calibrationCaptureReadiness, motionPrerequisite, motionFocusMismatch,
} from "./static/calibration.js";

test("motion prerequisites require an explicit camera reference and measured board confirmation", () => {
  assert.equal(motionPrerequisite("camera", "", false), "");
  assert.match(motionPrerequisite("imu", "", true), /camera-calibration reference/);
  assert.match(motionPrerequisite("imu", "camera", false), /measured printed board/);
  assert.equal(motionPrerequisite("imu", "camera", true), "");
});

test("motion preflight rejects the real focus change without a numeric tolerance", () => {
  const signature = { actual_camera2: { lens_focus_distance_diopters: 2.631579, active_physical_camera_id: "5" }, focus_build_fingerprint: "same-os" };
  const reference = { binding: { signature } };
  const focus = { mode: "manual_locked", physical_camera_id: "5", build_fingerprint: "same-os", focus_distance_diopters: Math.fround(2.631579) };
  assert.equal(motionFocusMismatch({ focus_control: focus }, reference), "");
  assert.match(motionFocusMismatch({ focus_control: { ...focus, focus_distance_diopters: 3.4364262 } }, reference), /does not match/);
  assert.match(motionFocusMismatch({ focus_control: { ...focus, focus_distance_diopters: Math.fround(2.631579) + 0.000001 } }, reference), /does not match/);
  for (const [key, value] of [["mode", "automatic"], ["physical_camera_id", "2"], ["build_fingerprint", "different"]]) assert.match(motionFocusMismatch({ focus_control: { ...focus, [key]: value } }, reference), /does not match/);
  assert.equal(motionFocusMismatch({}, reference), ""); // Unknown old snapshots still require the server binding check.
});

test("capture readiness names missing prerequisites without bypassing native readiness", () => {
  const native = { available: true, snapshot: { cameras: [], camera_index: -1 }, enabled: (action) => action === "checkPhone" };
  const missing = calibrationCaptureReadiness(native);
  assert.equal(missing.action, "checkPhone");
  assert.equal(missing.disabled, false);
  assert.match(missing.message, /not another recording/);
  assert.equal(calibrationCaptureReadiness({ ...native, enabled: () => false }).disabled, true);
  assert.equal(calibrationCaptureReadiness(null).disabled, true);
  assert.equal(calibrationCaptureReadiness({ ...native, configDirty: true }).action, "phone_setup");
  assert.equal(calibrationCaptureReadiness(native, "Confirm board geometry").action, "review_settings");
  assert.equal(calibrationCaptureReadiness({ ...native, snapshot: null }).action, "snapshot");
  assert.equal(calibrationCaptureReadiness({ ...native, bridge: { pendingAction: "checkPhone" } }).action, "snapshot");
  for (const key of ["active", "imu_active", "busy"]) {
    const busy = calibrationCaptureReadiness({ ...native, snapshot: { ...native.snapshot, [key]: true }, enabled: () => true });
    assert.equal(busy.disabled, true);
    assert.equal(busy.action, "none");
  }
  const cameras = [{ id: "same-camera" }];
  assert.equal(calibrationCaptureReadiness({ ...native, snapshot: { cameras, camera_index: -1 } }).action, "phone_setup");
  assert.equal(calibrationCaptureReadiness({ ...native, snapshot: { cameras, camera_index: 0 } }).action, "snapshot");
  const ready = calibrationCaptureReadiness({ ...native, snapshot: { cameras, camera_index: 0 }, enabled: () => true });
  assert.deepEqual(ready, { action: "capture", label: "", message: "", disabled: false });
});

test("editable defaults retain the exact nonlegacy ChArUco target", () => {
  assert.deepEqual(validateBoard(DEFAULT_BOARD), {
    squares_x: 10, squares_y: 14, square_length_m: 0.018, marker_length_m: 0.0132,
    dictionary: "DICT_4X4_1000", marker_ids: Array.from({ length: 70 }, (_, index) => index + 300), legacy_pattern: false,
  });
  assert.deepEqual(parseMarkerIds("300..302, 305; 310-311"), [300, 301, 302, 305, 310, 311]);
  assert.deepEqual(parseMarkerIds("369, 300, 305"), [369, 300, 305]);
});

test("short guide fixes the board and moves the phone in both camera stages", () => {
  assert.deepEqual(CALIBRATION_STEPS.map((step) => step.id), ["setup", "noise", "camera", "imu", "process", "profile", "walk"]);
  for (const mode of ["camera", "imu"]) {
    const step = CALIBRATION_STEPS.find((step) => step.id === mode);
    assert.match(step.text, /BOARD FIXED/);
    assert.match(step.text, /PHONE/);
  }
  assert.match(CALIBRATION_STEPS.find((step) => step.id === "noise").title, /60 seconds/);
  assert.doesNotMatch(JSON.stringify(CALIBRATION_STEPS), /3.hour|180.minute/i);
  assert.match(CALIBRATION_STEPS.find((step) => step.id === "camera").text, /staying near the distance where the board is sharp/);
  assert.match(CALIBRATION_STEPS.find((step) => step.id === "camera").text, /reduce movement if the board blurs/);
});

test("noise candidate is usable only from the explicit boolean, never from a status label or duration", () => {
  const candidate = { mode: "noise", status: "completed", result: { noise_model_usable: true, noise_model_status: "short_session_candidate", imu_noise_calibrated: false, accepted_for_metric_vio: false } };
  const outcome = calibrationOutcome(candidate);
  assert.match(outcome.title, /Short-session noise candidate/);
  assert.equal(outcome.gates.find((gate) => gate.key === "noise_model_usable").value, "Usable candidate");
  assert.equal(outcome.gates.find((gate) => gate.key === "imu_noise_calibrated").value, "Not established");
  assert.equal(outcome.gates.find((gate) => gate.key === "accepted_for_metric_vio").value, "Not accepted");
  assert.match(calibrationSummaryMarkup(candidate), /Long-term bias random walk remains unmeasured/);
  const coefficients = calibrationSummaryMarkup({ ...candidate, result: { ...candidate.result, noise: { gyroscope_random_walk: 0.001, gyroscope_noise_density: 0.002 }, noise_provenance: { gyroscope_random_walk: { measured: false, kind: "model_prior" }, gyroscope_noise_density: { measured: true } } } });
  assert.match(coefficients, /Gyroscope random walk · model prior, not measured/);
  assert.match(coefficients, /Gyroscope noise density · measured/);
  assert.doesNotMatch(coefficients, /Measured noise coefficients/);
  assert.doesNotMatch(calibrationOutcome({ ...candidate, result: { noise_model_status: "short_session_candidate", duration_s: 60 } }).title, /usable/);
});

test("insufficient-data reports pair human reasons with short corrective actions", () => {
  assert.match(calibrationNextAction("insufficient_image_coverage"), /board fixed.*move the phone/);
  assert.match(calibrationNextAction("heldout_lever_arm_unobservable", "imu"), /Translate the phone/);
  assert.match(calibrationNextAction("stationarity_not_supported", "noise"), /~60-second/);
  assert.match(calibrationNextAction("physical_board_scale_not_confirmed"), /Measure the printed/);
  assert.match(calibrationNextAction("gyroscope_x_random_walk_not_observable", "noise"), /not.*request for a long recording/);
  assert.match(calibrationNextAction("physical_sensor_row_geometry_unverified"), /more recording time alone will not repair/);
  const markup = calibrationSummaryMarkup({ mode: "camera", status: "completed", result: { quality: { status: "insufficient_evidence" }, reason_codes: ["<script>unknown</script>"] } });
  assert.match(markup, /What to do next/);
  assert.ok(!markup.includes("<script>"));
});

test("motion profile has an exact three-reference request and a separate readiness result", () => {
  const values = { camera_calibration_id: "cam", imu_calibration_id: "imu", noise_calibration_id: "noise" };
  assert.deepEqual(motionProfileRequest({ ...values, accepted_for_metric_vio: true }), values);
  for (const value of [null, 5, "", "x".repeat(161)]) assert.throws(() => motionProfileRequest({ ...values, imu_calibration_id: value }));
  const result = { mode: "motion_profile", status: "completed", result: { motion_profile_ready: true, accepted_for_metric_vio: false } };
  assert.match(calibrationOutcome(result).title, /ready for matching short walks/);
  assert.match(calibrationSummaryMarkup(result), /not a five-minute room-accuracy certificate/);
  assert.doesNotMatch(calibrationOutcome({ ...result, status: "running" }).title, /ready/);
});

test("board validation rejects unsafe, duplicate, reversed and oversized inputs without rewriting IDs", () => {
  for (const ids of ["300,300", "302..300", "0..100000000", "NaN", "-1", "1.5", "300<script>", []]) assert.throws(() => parseMarkerIds(ids));
  for (const board of [
    { ...DEFAULT_BOARD, squares_x: 1 }, { ...DEFAULT_BOARD, squares_y: 10.5 },
    { ...DEFAULT_BOARD, square_length_m: 0 }, { ...DEFAULT_BOARD, marker_length_m: Infinity },
    { ...DEFAULT_BOARD, marker_length_m: 0.02 }, { ...DEFAULT_BOARD, legacy_pattern: "false" },
    { ...DEFAULT_BOARD, dictionary: "<script>" },
  ]) assert.throws(() => validateBoard(board));
  assert.throws(() => validateBoard({ ...DEFAULT_BOARD, squares_x: 12 }), /exactly 84/);
  const board = validateBoard(DEFAULT_BOARD);
  assert.deepEqual(board.marker_ids, DEFAULT_BOARD.marker_ids);
  assert.notEqual(board.marker_ids, DEFAULT_BOARD.marker_ids);
});

test("job request preserves explicit mode, board confirmation and optional measured camera binding", () => {
  const request = calibrationRequest({ scan_id: "capture-1", mode: "imu", board: DEFAULT_BOARD, board_geometry_confirmed: false, camera_calibration_id: "camera-job-2", noise_calibration_id: "noise-job-3" });
  assert.equal(request.mode, "imu");
  assert.equal(request.schema, CALIBRATION_REQUEST_SCHEMA);
  assert.equal(request.board_geometry_confirmed, false);
  assert.equal(request.camera_calibration_id, "camera-job-2");
  assert.equal(request.noise_calibration_id, "noise-job-3");
  assert.throws(() => calibrationRequest({ ...request, noise_calibration_id: 3 }), /valid measured calibration/);
  assert.equal(Object.hasOwn(request, "metric_vio_allowed"), false);
  assert.throws(() => calibrationRequest({ ...request, mode: "walk" }), /camera intrinsics/);
  assert.throws(() => calibrationRequest({ ...request, board_geometry_confirmed: "true" }), /whether the printed/);
});

test("native tagged intent is distinct from existing calibration admission fields", () => {
  const request = { mode: "camera", board: DEFAULT_BOARD, board_geometry_confirmed: false };
  assert.deepEqual(selectedCalibrationRequest({ calibration_request: request }), request);
  assert.deepEqual(selectedCalibrationRequest({ selected_capture: { calibration_request: request } }), request);
  assert.equal(selectedCalibrationRequest({ selected_capture: { calibration: request } }), null);
  assert.equal(selectedCalibrationRequest({ calibration_request: {} }), null);
  assert.equal(selectedCalibrationRequest({ selected_capture: "selected", calibration_request: { ...request, capture_id: "different" } }), null);
  assert.deepEqual(selectedCalibrationRequest({ selected_capture: "selected", calibration_request: { ...request, capture_id: "selected" } }), { ...request, capture_id: "selected" });
});

test("board limits match the current backend, including dimensions, topology and dictionary capacity", () => {
  const maximum = { ...DEFAULT_BOARD, squares_x: 30, squares_y: 30, square_length_m: 0.1, marker_length_m: 0.001, marker_ids: Array.from({ length: 450 }, (_, i) => i) };
  assert.equal(validateBoard(maximum).squares_x, 30);
  for (const board of [{ ...maximum, square_length_m: 0.10001 }, { ...DEFAULT_BOARD, square_length_m: 0.201 }, { ...DEFAULT_BOARD, marker_length_m: 0.0009 }, { ...DEFAULT_BOARD, squares_x: 2 }, { ...DEFAULT_BOARD, squares_y: 31 }, { ...DEFAULT_BOARD, dictionary: "DICT_4X5_1000" }, { ...DEFAULT_BOARD, dictionary: "DICT_4X4_50" }]) assert.throws(() => validateBoard(board));
});

test("calibration summaries distinguish completion, qualified intrinsics, noise, and metric acceptance", () => {
  const job = { mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true, imu_noise_calibrated: false, accepted_for_metric_vio: false, quality: { status: "qualified" }, camera_result: { model: "opencv_pinhole", training_rms_px: 0.8, resolution: [7680, 4320], K: [[5000, 0, 3840], [0, 5001, 2160], [0, 0, 1]], D: [0.1, -0.01] } } };
  const outcome = calibrationOutcome(job);
  assert.match(outcome.title, /Camera fit qualified/);
  assert.equal(outcome.gates.find((row) => row.key === "accepted_for_metric_vio").value, "Not accepted");
  assert.equal(outcome.gates.find((row) => row.key === "time_offset_calibrated").value, "Not reported");
  assert.match(calibrationSummaryMarkup(job), /5,000 \/ 5,001 px/);
  assert.match(calibrationSummaryMarkup(job), /0.8 px/);
  assert.match(calibrationOutcome({ ...job, mode: "noise", result: { imu_noise_calibrated: true, accepted_for_metric_vio: false } }).title, /IMU noise qualified/);
  assert.match(calibrationOutcome({ status: "completed" }).title, /review qualification/);
  const unsafe = calibrationSummaryMarkup({ status: "failed", result: { error: "<script>alert(1)</script>", reason_codes: ["insufficient_image_coverage"], quality: { status: "insufficient_evidence" } } });
  assert.ok(!unsafe.includes("<script>"));
  assert.match(unsafe, /Insufficient image coverage/);
});

test("job cards escape all server strings and report links stay on the configured HTTP(S) origin", () => {
  const markup = calibrationJobCards([{ id: 'job" onclick="alert(1)', scan_id: "<img src=x>", status: "<script>", mode: "camera" }], "");
  assert.ok(!markup.includes("<img"));
  assert.ok(!markup.includes("<script>"));
  assert.ok(markup.includes("&quot;"));
  const origin = "https://roomwalk.test:8789";
  assert.equal(safeReportUrl("/assets/report.json", origin), `${origin}/assets/report.json`);
  for (const value of ["javascript:alert(1)", "file:///private", "data:text/html,<script>", "https://elsewhere.test/report", "https://user:pass@roomwalk.test:8789/report", "http://roomwalk.test:8789/report"]) assert.equal(safeReportUrl(value, origin), null);
});

test("calibration client forwards explicit jobs requests and handles a not-yet-deployed backend", async () => {
  const calls = [];
  const result = await calibrationJsonFetch("/api/calibration/jobs", { method: "POST", body: "{}" }, {
    fetchImpl: async (url, options) => { calls.push([url, options]); return { ok: true, json: async () => ({ id: "job-1", status: "queued" }) }; },
  });
  assert.equal(result.id, "job-1");
  assert.equal(calls[0][1].method, "POST");
  assert.ok(calls[0][1].signal instanceof AbortSignal);
  await assert.rejects(calibrationJsonFetch("/api/calibration/jobs", {}, { fetchImpl: async () => ({ ok: false, status: 404 }) }), /Native capture and board editing still work/);
  await assert.rejects(calibrationJsonFetch("/api/calibration/jobs/missing", {}, { fetchImpl: async () => ({ ok: false, status: 404, json: async () => ({ detail: "Referenced calibration was not found" }) }) }), /Referenced calibration was not found/);
});

test("calibration status and response-body waits are bounded", async () => {
  let signal;
  await assert.rejects(calibrationJsonFetch("/api/calibration/jobs", {}, {
    timeoutMs: 10,
    fetchImpl: async (_url, options) => { signal = options.signal; return { ok: true, json: () => new Promise(() => {}) }; },
  }), /did not respond in time/);
  assert.equal(signal.aborted, true);
});

test("scan profile status uses the root computed result, never the immutable import selection", () => {
  const intent = { status: "applied", profile_id: "selected-at-import" };
  assert.equal(scanCameraSelectionMarkup({ capture: { camera_calibration_selection: intent } }), "");
  const mismatch = scanCameraSelectionMarkup({ capture: { camera_calibration_selection: intent }, camera_calibration_selection: { status: "not_applied", profile_id: "reviewed-profile", reason_codes: ["locked_focus_mismatch", "<img src=x>"], accepted_for_metric_vio: false } });
  assert.match(mismatch, /Selected camera profile not applied/);
  assert.match(mismatch, /locked_focus_mismatch/);
  assert.ok(!mismatch.includes("<img"));
  assert.ok(!mismatch.includes("selected-at-import"));
  assert.match(scanCameraSelectionMarkup({ camera_calibration_selection: { status: "applied", accepted_for_metric_vio: false } }), /Camera profile applied to this scan/);
});

test("scan motion status uses only computed binding and offers explicit guarded recheck", () => {
  const selection = { motion_calibration_id: "checked-job", profile_id: "candidate" };
  assert.equal(scanMotionProfileMarkup({ capture: { motion_calibration_selection: selection } }), "");
  const scan = { status: "ready", prepared: { frames: [] }, capture: { motion_calibration_selection: selection, metric_vio_allowed: false, motion_profile: { status: "not_applied", message: "<script>bad</script> Lens mismatch", reason_codes: ["motion_profile_verification_failed"] } } };
  const markup = scanCameraSelectionMarkup(scan);
  assert.match(markup, /Motion profile not applied · RGB remains available/);
  assert.match(markup, /Recheck profile and run OpenVINS/);
  assert.ok(!markup.includes("<script>"));
  assert.doesNotMatch(scanMotionProfileMarkup({ ...scan, vio: { status: "running" } }), /id="initiate-vio"/);
  assert.doesNotMatch(scanMotionProfileMarkup({ ...scan, capture: { ...scan.capture, motion_calibration_selection: null } }), /id="initiate-vio"/);
  const applied = scanMotionProfileMarkup({ ...scan, capture: { ...scan.capture, metric_vio_allowed: true, motion_profile: { status: "applied", profile_id: "checked-profile" } } });
  assert.match(applied, /Motion profile bound to this walk/);
  assert.match(applied, /not a room-accuracy certificate/);
  assert.doesNotMatch(applied, /id="initiate-vio"/);
});
