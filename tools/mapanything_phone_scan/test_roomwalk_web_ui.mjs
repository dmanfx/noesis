// Optional real-DOM smoke, with all HTTP responses intercepted in-process.
// No server, native service, camera, solver, or GPU is started by this test.
import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { DEFAULT_BOARD } from "./static/calibration.js";
let chromium;
try { ({ chromium } = createRequire(import.meta.url)(process.env.ROOMWALK_PLAYWRIGHT_MODULE || "playwright")); }
catch (_) { /* Dependency is optional for the standalone CPU controller lane. */ }

async function fixture(t, { native = true, offline = false, scans = [], health = {}, expandDetails = true, noiseResult = { noise_model_usable: true, noise_model_status: "short_session_candidate", imu_noise_calibrated: false } } = {}) {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.ROOMWALK_CHROMIUM_EXECUTABLE || chromium.executablePath(), args: ["--disable-gpu"] });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 412, height: 915 }, userAgent: native ? "Mozilla/5.0 RoomWalkAndroid/0.2" : "Mozilla/5.0 Chrome" });
  const requests = [], errors = [], jobs = [];
  let cameraSelection = null;
  let motionSelection = null;
  page.on("pageerror", (error) => errors.push(error.message));
  await page.addInitScript(() => { window.__nativeCommands = []; });
  await page.route("**/*", async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    requests.push({ path: pathname, url: request.url(), method: request.method(), body: request.postDataJSON() });
    const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (pathname === "/" || pathname.startsWith("/static/")) {
      const filename = pathname === "/" ? "index.html" : pathname.slice("/static/".length);
      if (!["index.html", "app.js", "style.css", "browser_capture.js", "companion_capture.js", "native_capture.js", "calibration.js", "path_comparison.js"].includes(filename)) return route.abort();
      let source = await readFile(new URL(`./static/${filename}`, import.meta.url), "utf8");
      if (filename === "native_capture.js") source += '\nconst realSend = NativeBridge.prototype.send; NativeBridge.prototype.send = function(action, args = {}) { this.navigate = (url) => window.__nativeCommands.push(JSON.parse(new URL(url).searchParams.get("request"))); return realSend.call(this, action, args); };\n';
      return route.fulfill({ contentType: filename.endsWith(".js") ? "text/javascript" : filename.endsWith(".css") ? "text/css" : "text/html", body: source });
    }
    if (offline || pathname.startsWith("/three/")) return route.abort();
    if (pathname === "/api/health") return json({ device: "CPU fixture", max_selected_frames: 256, alignment_targets: [], ...health });
    if (pathname === "/api/scans") return json(scans);
    if (pathname.endsWith("/path-review") && request.method() === "POST") {
      const scan = scans.find((scan) => pathname === `/api/scans/${scan.id}/path-review`);
      if (!scan || scan.status !== "complete") return json({ detail: "Run the visual model first" }, 409);
      scan.path_review = { status: "queued", progress: 0, message: "Preparing retained path review" };
      return json(scan, 202);
    }
    if (pathname.endsWith("/supplements/from-scan") && request.method() === "POST") {
      const scan = scans.find((scan) => pathname === `/api/scans/${scan.id}/supplements/from-scan`);
      scan.supplements = [{ id: "addition-1", status: "processing_frames", source_capture: { scan_id: new URL(request.url()).searchParams.get("source_scan_id") } }];
      return json(scan, 202);
    }
    if (pathname.endsWith("/initiate-vio")) {
      const scan = scans.find((scan) => pathname === `/api/scans/${scan.id}/initiate-vio`);
      if (scan) { scan.vio = { status: "queued", message: "Rechecking motion profile" }; return json(scan, 202); }
    }
    if (pathname.startsWith("/api/scans/")) return json(scans.find((scan) => pathname === `/api/scans/${scan.id}`) || {}, 200);
    if (pathname === "/api/calibration/camera-selection") {
      if (request.method() === "POST") {
        const id = request.postDataJSON().camera_calibration_id;
        cameraSelection = id === null ? null : { camera_calibration_id: id, profile_id: "measured-camera-profile", selected_at: "2026-09-14T15:00:00Z", binding_sha256: "abc123" };
      }
      return json({ selection: cameraSelection });
    }
    if (pathname === "/api/calibration/motion-selection") {
      if (request.method() === "POST") {
        const id = request.postDataJSON().motion_calibration_id;
        motionSelection = id === null ? null : { motion_calibration_id: id, profile_id: "checked-motion-profile", status: "ready_for_short_walks", maximum_duration_s: 300 };
      }
      return json({ selection: motionSelection });
    }
    if (pathname === "/api/calibration/motion-profile-jobs") {
      const job = { id: "motion-check", mode: "motion_profile", request: request.postDataJSON(), status: "queued", result: { motion_profile_ready: false }, artifacts: { motion_profile_url: "/profiles/motion.json", validation_url: "/reports/motion-validation.json" } };
      jobs.push(job); return json({ job }, 202);
    }
    if (pathname === "/api/calibration/imu-recordings") return json({ recordings: [{ capture_id: "imu-short" }] });
    if (pathname === "/api/calibration/noise-jobs") {
      const job = { id: "noise-job", mode: "noise", status: "completed", request: request.postDataJSON(), result: noiseResult };
      jobs.push(job); return json(job, 202);
    }
    if (pathname === "/api/calibration/jobs") {
      if (request.method() === "POST") { const payload = request.postDataJSON(); if (payload.schema !== "roomwalk.calibration_request.v1") return json({ detail: "request must use roomwalk.calibration_request.v1" }, 422); const job = { id: "job-1", ...payload, request: payload, status: "queued", artifacts: { report_url: "/reports/job.json", unsafe_url: "javascript:alert(1)" } }; jobs.push(job); return json(job, 202); }
      return json({ jobs });
    }
    if (pathname.endsWith("/cancel")) { jobs[0].status = "cancelled"; return json(jobs[0]); }
    if (pathname.endsWith("noise_allan.svg")) return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320" viewBox="0 0 800 320"><rect width="800" height="320" fill="#f8faf9"/><text x="40" y="35" fill="#29423a" font-size="20">Allan deviation · mocked UI fixture</text><path d="M60 65V265H750" fill="none" stroke="#567"/><path d="M70 80L170 145 300 200 430 225 580 205 740 150" fill="none" stroke="#19786d" stroke-width="3"/><path d="M70 95L170 155 300 210 430 230 580 215 740 160" fill="none" stroke="#4568ae" stroke-width="3" stroke-dasharray="8 6"/><text x="280" y="300" fill="#345" font-size="16">Averaging time (s)</text></svg>' });
    if (pathname.startsWith("/api/calibration/jobs/")) return json(jobs.find((job) => pathname.endsWith(`/${job.id}`)) || {}, 200);
    return route.abort();
  });
  await page.goto("https://roomwalk.test/");
  await page.waitForFunction(() => typeof window.RoomWalkNative?.receive === "function");
  // Legacy control-specific checks explicitly expand the advanced workspace.
  // Workflow tests leave these closed and use only the guided primary action.
  if (expandDetails) await page.evaluate(() => {
    for (const id of ["calibration-settings", "calibration-noise-step", "calibration-profile-details", "calibration-history", "calibration-step-picker"]) document.getElementById(id).open = true;
  });
  const emit = (fields = {}) => page.evaluate((fields) => window.RoomWalkNative.receive({
    schema: "roomwalk.native_state.v1", server: "https://roomwalk.test", status: "Ready", detail: "Native fixture",
    cameras: [{ label: "Rear main", full_walk_available: true, test_available: true }], camera_index: 0,
    room_cameras: [{ camera_id: "kitchen", label: "Kitchen" }], room_camera_index: 0, saved: [],
    enabled: { configure: true, checkPhone: true, checkConnection: true, capture: true, startImu: true, stopImu: true, upload: true, selectSaved: true }, ...fields,
  }), fields);
  return { page, requests, errors, emit, jobs };
}

async function checkLayout(page, label) {
  for (const [width, height] of [[360, 640], [412, 915], [1280, 900]]) {
    await page.setViewportSize({ width, height });
    await page.evaluate(() => scrollTo(0, 0));
    if (process.env.ROOMWALK_UI_SCREENSHOT_DIR) await page.screenshot({ path: join(process.env.ROOMWALK_UI_SCREENSHOT_DIR, `${label}-${width}.png`), fullPage: true });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth ? [] :
      Array.from(document.querySelectorAll("body *")).filter((element) => {
        const rect = element.getBoundingClientRect(); return rect.width > 0 && rect.right > innerWidth + 1;
      }).slice(0, 20).map((element) => element.tagName + "." + element.className + "#" + element.id));
    assert.deepEqual(overflow, [], `${label} fits ${width}px`);
  }
  await page.setViewportSize({ width: 412, height: 915 });
}

async function openSetup(page) {
  await page.getByRole("tab", { name: "Setup", exact: true }).click();
  if (!(await page.locator("#calibration-secondary").evaluate((element) => element.open))) {
    await page.locator("#calibration-secondary > summary").click();
  }
}

const walkIntent = (mode = "reconstruction", target = null) => ({
  schema: "roomwalk.walk_intent.v1", mode, target_scan_id: target,
  carry_protocol: mode === "path_refinement" ? "close_body" : "coverage", accuracy_target_m: 0.1,
});
const reconstruction = (id = "room") => ({
  id, name: id, status: "complete", created_at: "2026-09-19T12:00:00Z",
  outputs: { view_count: 12, review_point_count: 1000, artifact_urls: {} },
});

test("native shared purpose controls stay visible with browser details closed and filter invalid targets", { skip: !chromium }, async (t) => {
  const scans = [
    reconstruction(), { ...reconstruction("path"), walk_intent: walkIntent("path_refinement", "room") },
    { ...reconstruction("calibration"), capture: { calibration_request: { mode: "camera" } } },
    { ...reconstruction("no-output"), outputs: null }, { ...reconstruction("pending"), status: "ready" },
    { ...reconstruction("nested-path"), capture: { walk_intent: walkIntent("path_refinement", "room") } },
  ];
  const { page, emit, errors } = await fixture(t, { scans, expandDetails: false });
  await emit();
  assert.equal(await page.locator("#browser-tools").getAttribute("open"), null);
  for (const id of ["capture-mode-reconstruction", "capture-mode-path", "reconstruction-target-scan"]) {
    assert.equal(await page.locator("#" + id).isVisible(), true);
    assert.equal(await page.locator("#" + id).evaluate((element) => element.closest("#browser-tools")), null);
  }
  await page.waitForFunction(() => document.querySelector("#reconstruction-target-scan").options.length === 2);
  assert.deepEqual(await page.locator("#reconstruction-target-scan option").evaluateAll((options) => options.map((o) => o.value)), ["", "room"]);
  await checkLayout(page, "shared-native-reconstruction");
  await page.locator("#capture-mode-path").check();
  assert.equal(await page.locator("#path-target-scan").isVisible(), true);
  assert.equal(await page.locator("#path-camera-select").isVisible(), true);
  assert.equal(await page.locator('input[name="path-direction"]').count(), 0);
  assert.deepEqual(await page.locator("#path-target-scan option").evaluateAll((options) => options.map((o) => o.value)), ["", "room"]);
  assert.match(await page.locator("#path-options").innerText(), /your own torso.*elbows tucked/s);
  await checkLayout(page, "shared-native-path");
  await page.getByRole("tab", { name: "Setup", exact: true }).click();
  assert.equal(await page.locator("#calibration-secondary").getAttribute("open"), null);
  assert.equal(await page.locator("#calibration-test").isVisible(), false);
  assert.deepEqual(errors, []);
});

test("native path requires target and acknowledged paired camera and sends the exact intent", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t, { scans: [reconstruction()], expandDetails: false });
  const room_cameras = [{ camera_id: "kitchen", label: "Kitchen" }, { camera_id: "family", label: "Family Room" }];
  await emit({ room_cameras });
  await page.locator("#capture-mode-path").check();
  await page.locator("#native-capture-primary").click();
  assert.match(await page.locator("#toast").innerText(), /Choose the existing reconstruction/);
  await page.locator("#path-target-scan").selectOption("room");
  await page.locator("#native-capture-primary").click();
  assert.match(await page.locator("#toast").innerText(), /Choose the paired Noesis camera/);
  assert.equal((await page.evaluate(() => window.__nativeCommands)).some((c) => c.action === "capture"), false);
  await page.locator("#path-camera-select").selectOption("family");
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "configure", args: { room_camera_index: 1 } });
  assert.equal(await page.locator("#native-capture-primary").isDisabled(), true);
  await emit({ room_cameras, room_camera_index: 0, walk_intent: walkIntent() });
  await page.locator("#native-capture-primary").click();
  assert.match(await page.locator("#toast").innerText(), /confirm the selected paired camera/);
  await emit({ room_cameras, room_camera_index: 1, walk_intent: walkIntent() });
  assert.equal(await page.locator("#capture-mode-path").isChecked(), true);
  await page.locator("#native-capture-primary").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), {
    action: "capture", args: { mode: "path_refinement", target_scan_id: "room", carry_protocol: "close_body", accuracy_target_m: 0.1 },
  });
  const captureCount = () => page.evaluate(() => window.__nativeCommands.filter((command) => command.action === "capture").length);
  // A later native setup change cannot make the retained dropdown authorize a
  // different actual camera. Snapshot delivery alone must never start capture.
  await emit({ room_cameras, room_camera_index: 0, walk_intent: walkIntent("path_refinement", "room") });
  assert.equal(await page.locator("#path-camera-select").inputValue(), "family");
  assert.equal(await captureCount(), 1);
  await page.locator("#native-capture-primary").click();
  assert.match(await page.locator("#toast").innerText(), /Select and apply the same camera/);
  assert.equal(await captureCount(), 1);
  await emit({ room_cameras, room_camera_index: -1 });
  await page.locator("#native-capture-primary").click();
  assert.equal(await captureCount(), 1, "Phone-only native setup cannot start a path capture");
  // Match by camera identity even if the native inventory order changes.
  await emit({ room_cameras: [...room_cameras].reverse(), room_camera_index: 0 });
  assert.equal(await captureCount(), 1, "A matching acknowledgement does not auto-capture");
  await page.locator("#native-capture-primary").click();
  assert.equal(await captureCount(), 2);
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), {
    action: "capture", args: { mode: "path_refinement", target_scan_id: "room", carry_protocol: "close_body", accuracy_target_m: 0.1 },
  });
  assert.deepEqual(errors, []);
});

test("native reconstruction sends its selected add-views target and Library adds it explicitly", { skip: !chromium }, async (t) => {
  const source = { id: "extra", name: "Extra views", status: "ready", walk_intent: walkIntent("reconstruction", "room") };
  const { page, emit, requests, errors } = await fixture(t, { scans: [reconstruction(), source], expandDetails: false });
  await emit();
  await page.locator("#reconstruction-target-scan").selectOption("room");
  await page.locator("#native-capture-primary").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), {
    action: "capture", args: { mode: "reconstruction", target_scan_id: "room", carry_protocol: "coverage", accuracy_target_m: 0.1 },
  });
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="extra"]').click();
  assert.equal(requests.some((r) => r.path.endsWith("/supplements/from-scan")), false);
  await page.locator("#add-views-to-target").click();
  await page.waitForFunction(() => document.querySelector("#toast")?.textContent.includes("explicit additive revision"));
  const request = requests.find((r) => r.path.endsWith("/supplements/from-scan"));
  assert.equal(request.method, "POST");
  assert.equal(new URL(request.url).searchParams.get("source_scan_id"), "extra");
  assert.deepEqual(errors, []);
});

test("native reconstruction supports explicit phone-only capture with a populated room-camera inventory", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t, { expandDetails: false });
  await emit({ room_camera_index: -1 });
  assert.equal(await page.locator("#native-setup").getAttribute("open"), null);
  assert.equal(await page.locator('#native-room-camera option[value="-1"]').textContent(), "None · phone-only reconstruction");
  assert.equal(await page.locator("#native-room-camera").inputValue(), "-1");
  await page.locator("#native-capture-primary").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), {
    action: "capture", args: { mode: "reconstruction", carry_protocol: "coverage", accuracy_target_m: 0.1 },
  });
  await emit({ room_camera_index: 0 });
  await page.locator("#native-setup > summary").click();
  await page.locator("#native-room-camera").selectOption("-1");
  await page.locator('#native-configuration button[type="submit"]').click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).args.room_camera_index, -1);
  await emit({ room_camera_index: -1, imu_minutes: 1 });
  await page.locator("#native-capture-primary").click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).args.mode, "reconstruction");
  assert.deepEqual(errors, []);
});

test("optional camera and sensor setup work with unfinished path selection and preserve the walk mode", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t);
  await emit();
  await page.locator("#capture-mode-path").check();
  await openSetup(page);
  await page.locator("#calibration-test").click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).args.mode, "camera");
  await emit();
  await page.locator("#noise-start").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "startImu", args: { minutes: 1 } });
  await emit();
  await page.getByRole("tab", { name: "Capture", exact: true }).click();
  assert.equal(await page.locator("#capture-mode-path").isChecked(), true);
  assert.match(await page.locator("#native-intent-summary").innerText(), /Path refinement/);
  await page.locator("#native-capture-primary").click();
  assert.match(await page.locator("#toast").innerText(), /Choose the existing reconstruction/);
  assert.deepEqual(errors, []);
});

test("legacy evidence stays available and path review polls to completed artifact links without qualifying accuracy", { skip: !chromium }, async (t) => {
  const legacy = { id: "legacy", name: "Legacy capture", status: "ready", video: { url: "/files/legacy.mp4", size_bytes: 1000 } };
  const path = { id: "path", name: "Path walk", status: "ready", walk_intent: walkIntent("path_refinement", "room") };
  const { page, emit, requests, errors } = await fixture(t, { scans: [reconstruction(), legacy, path], expandDetails: false });
  await emit();
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="legacy"]').click();
  assert.match(await page.locator(".legacy-intent").innerText(), /intent not recorded/);
  assert.equal(await page.getByRole("link", { name: "Original video", exact: true }).getAttribute("href"), "/files/legacy.mp4");
  assert.equal(await page.locator("#initiate-inference").isVisible(), true);
  await checkLayout(page, "legacy-evidence");
  await page.locator('[data-scan-id="path"]').click();
  assert.equal(await page.locator("#review-path").isDisabled(), true);
  assert.match(await page.locator(".path-review-card").innerText(), /Run reconstruction below/);
  path.status = "complete"; path.outputs = reconstruction().outputs;
  await page.waitForFunction(() => document.querySelector("#review-path")?.disabled === false);
  assert.equal(await page.locator("#additional-camera-video").count(), 0);
  assert.equal(await page.getByText("Additional room views", { exact: true }).count(), 0);
  await checkLayout(page, "path-review-ready");
  await page.locator("#review-path").click();
  await page.waitForFunction(() => document.querySelector("#review-path") === null);
  assert.equal(await page.locator("#delete-scan").count(), 0);
  const reviewRequest = requests.find((r) => r.path === "/api/scans/path/path-review");
  assert.equal(new URL(reviewRequest.url).searchParams.get("target_scan_id"), "room");
  path.path_review.status = "running"; path.path_review.progress = 0.5; path.path_review.message = "Checking retained evidence";
  await page.waitForFunction(() => document.querySelector(".path-review-card")?.textContent.includes("50%"));
  assert.equal(await page.locator("#delete-scan").count(), 0);
  path.path_review = { status: "complete", results: {
    artifact_url: "/files/path-review.json",
    artifact_urls: { report: "/files/report.json", trajectory: "/files/trajectory.json", motion: "/files/motion.json", motion_intervals: "/files/intervals.json" },
    visual_path: { status: "available", pose_count: 24 }, imu_consistency: { status: "available" },
    accuracy: { target_m: 0.1, qualified: false, measured_position_error_m: null }, body_ground_reference: { status: "not_established" },
  } };
  await page.getByRole("link", { name: "Export path review" }).waitFor();
  assert.equal(await page.getByRole("link", { name: "Export path review" }).getAttribute("href"), "/files/path-review.json");
  for (const name of ["Review report", "Visual trajectory", "IMU consistency", "Motion intervals"]) {
    assert.equal(await page.getByRole("link", { name, exact: true }).isVisible(), true);
  }
  assert.match(await page.locator(".path-review-card").innerText(), /Qualified: no.*Measured position error: not reported/s);
  assert.equal(await page.locator("#delete-scan").isVisible(), true);
  assert.equal(requests.filter((r) => r.path.endsWith("/path-review") && r.method === "POST").length, 1);
  await checkLayout(page, "path-review-complete");
  assert.deepEqual(errors, []);
});

test("legacy native path review uses self-reference and leaves the unknown carry protocol unchanged", { skip: !chromium }, async (t) => {
  const legacy = { id: "legacy-native", name: "Old native walk", status: "ready", capture: { capture_kind: "android_camera_imu" } };
  const calibration = { ...reconstruction("calibration"), capture: { capture_kind: "android_camera_imu", calibration_request: { mode: "camera" } } };
  const { page, emit, requests, errors } = await fixture(t, { scans: [legacy, calibration], expandDetails: false });
  await emit();
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="legacy-native"]').click();
  assert.match(await page.locator(".path-review-card").innerText(), /Retained camera path; original carry protocol unknown/);
  assert.equal(await page.locator("#review-path").isDisabled(), true);
  legacy.status = "complete";
  await page.locator("#refresh-button").click();
  assert.equal(await page.locator("#review-path").isDisabled(), true, "Completion without outputs cannot start review");
  legacy.outputs = reconstruction().outputs;
  await page.waitForFunction(() => document.querySelector("#review-path")?.disabled === false);
  await page.locator("#review-path").click();
  await page.waitForFunction(() => document.querySelector("#review-path") === null);
  const request = requests.find((r) => r.path === "/api/scans/legacy-native/path-review");
  assert.equal(new URL(request.url).search, "", "No guessed target is sent for a legacy self-reference review");
  legacy.path_review = { status: "complete", results: { artifact_url: "/files/legacy-review.json", accuracy: { target_m: 0.1, qualified: false } } };
  await page.getByRole("link", { name: "Export path review" }).waitFor();
  assert.match(await page.locator(".path-review-card").innerText(), /original carry protocol unknown/);
  assert.match(await page.locator(".legacy-intent").innerText(), /intent not recorded/);
  assert.doesNotMatch(await page.locator(".path-review-card").innerText(), /Self-carried|elbows tucked|close-body/);
  assert.equal(Object.hasOwn(legacy, "walk_intent"), false);
  assert.equal(requests.some((r) => r.path.endsWith("/walk-intent") && r.method === "POST"), false);
  await checkLayout(page, "legacy-native-path-review");
  await page.locator('[data-scan-id="calibration"]').click();
  assert.equal(await page.locator(".path-review-card").count(), 0);
  assert.deepEqual(errors, []);
});

test("offline Android boots native tools and calibration without 3D dependencies", { skip: !chromium }, async (t) => {
  const { page, requests, errors, emit } = await fixture(t, { offline: true });
  await emit();
  assert.equal(await page.locator("#native-capture-panel").isVisible(), true);
  assert.ok((await page.evaluate(() => window.__nativeCommands)).some((command) => command.action === "snapshot"));
  await checkLayout(page, "native-capture");
  await openSetup(page);
  await page.locator("#calibration-test").click();
  const command = (await page.evaluate(() => window.__nativeCommands)).at(-1);
  assert.equal(command.action, "capture");
  assert.equal(command.args.mode, "camera");
  assert.deepEqual(command.args.board, DEFAULT_BOARD);
  assert.equal(command.args.board_geometry_confirmed, false);
  await checkLayout(page, "camera-calibration");
  assert.equal(requests.some((request) => request.path.startsWith("/three/")), false);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("tagged captures hydrate calibration, explicit processing/cancellation preserves reports and source", { skip: !chromium }, async (t) => {
  const intent = { schema: "roomwalk.calibration_request.v1", mode: "imu", board: { ...DEFAULT_BOARD, square_length_m: 0.02 }, board_geometry_confirmed: true, camera_calibration_id: "camera-2" };
  const scan = { id: "scan-1", name: '<img src=x onerror="alert(1)">', status: "calibration_ready", capture: { calibration_request: intent }, video: { size_bytes: 42 } };
  const { page, requests, errors, emit } = await fixture(t, { scans: [scan] });
  await emit({ selected_capture: "take-1", calibration_request: intent });
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="scan-1"]').click();
  await page.locator("#open-calibration").click();
  await openSetup(page);
  assert.equal(await page.locator("#calibration-mode").inputValue(), "imu");
  assert.equal(await page.locator("#board-square-length").inputValue(), "0.02");
  assert.equal(await page.locator("#board-confirmed").isChecked(), true);
  assert.equal(await page.locator("#scan-list img").count(), 0);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  await page.locator("#calibration-process").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail")?.textContent.includes("queued"));
  const submitted = requests.find((request) => request.path === "/api/calibration/jobs" && request.method === "POST").body;
  assert.equal(submitted.scan_id, "scan-1");
  assert.equal(submitted.camera_calibration_id, "camera-2");
  assert.equal(await page.locator('#calibration-job-detail a[href^="javascript:"]').count(), 0);
  page.on("dialog", (dialog) => dialog.accept());
  await page.locator("[data-calibration-cancel]").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail")?.textContent.includes("cancelled"));
  assert.equal(requests.some((request) => request.path.includes("initiate-")), false);
  assert.deepEqual(errors, []);
});

test("stationary noise starts one minute without a long-protocol CTA, then processes an explicit retained take", { skip: !chromium }, async (t) => {
  const { page, requests, errors, emit } = await fixture(t);
  await emit();
  await openSetup(page);
  assert.equal(await page.locator("#noise-start").isVisible(), true);
  assert.equal(await page.locator("#calibration-mode").inputValue(), "camera");
  assert.doesNotMatch(await page.locator("#calibration-panel").innerText(), /3.hour|180.minute|minimum.*hours/i);
  await page.locator("#noise-start").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "startImu", args: { minutes: 1 } });
  await page.locator("#noise-recording").selectOption("imu-short");
  await page.locator("#noise-process").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail")?.textContent.includes("noise-job"));
  assert.deepEqual(requests.find((request) => request.path === "/api/calibration/noise-jobs").body, { imu_capture_id: "imu-short" });
  assert.match(await page.locator("#calibration-job-detail").innerText(), /Short-session noise candidate/);
  assert.equal(await page.locator("[data-calibration-measurements]").getAttribute("open"), null);
  assert.match(await page.locator("[data-calibration-measurements]").textContent(), /Long-term bias random walk remains unmeasured/);
  await page.locator('[data-calibration-reference="noise-job"]').click();
  assert.equal(await page.locator("#calibration-noise-id").inputValue(), "noise-job");
  assert.equal(await page.locator('[data-calibration-step="camera"]').getAttribute("aria-current"), "step");
  assert.equal(await page.locator("#native-imu-minutes").getAttribute("max"), "5");
  await emit({ imu_minutes: "180" });
  assert.equal(await page.locator("#native-imu-minutes").inputValue(), "1");
  await checkLayout(page, "imu-calibration");
  assert.deepEqual(errors, []);
});

test("upload identity hydrates the exact tagged scan and refreshes never overwrite edits", { skip: !chromium }, async (t) => {
  const intent = { mode: "camera", board: DEFAULT_BOARD, board_geometry_confirmed: true };
  const scan = { id: "received-scan", status: "calibration_ready", capture: { calibration_request: intent } };
  const { page, requests, emit, errors } = await fixture(t, { scans: [scan] });
  await emit();
  await emit({ last_scan_id: "received-scan", selected_capture: "phone-take", calibration_request: intent });
  await page.locator("#panel-setup").waitFor({ state: "visible" });
  await openSetup(page);
  assert.equal(await page.locator("#calibration-scan").inputValue(), "received-scan");
  await page.locator("#calibration-board-settings summary").click();
  await page.locator("#board-square-length").fill("0.022");
  await emit({ last_scan_id: "received-scan", selected_capture: "phone-take", calibration_request: intent });
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator("#refresh-button").click();
  await openSetup(page);
  assert.equal(await page.locator("#board-square-length").inputValue(), "0.022");
  assert.equal(await page.locator("#board-confirmed").isChecked(), false);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("malformed tagged settings cannot inherit an old board or submit a default job", { skip: !chromium }, async (t) => {
  const scan = { id: "bad-board", status: "calibration_ready", capture: { calibration_request: { mode: "camera", board: { squares_x: 12 }, board_geometry_confirmed: true } } };
  const { page, requests, emit, errors } = await fixture(t, { scans: [scan] });
  await emit();
  await openSetup(page);
  await page.locator("#calibration-scan").selectOption("bad-board");
  assert.equal(await page.locator("#board-square-length").inputValue(), "");
  assert.equal(await page.locator("#board-confirmed").isChecked(), false);
  assert.equal(await page.locator("#calibration-process").isDisabled(), true);
  assert.equal(await page.locator("#calibration-capture").isDisabled(), true);
  await page.locator("#calibration-defaults").click();
  assert.equal(await page.locator("#calibration-process").isDisabled(), false);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("completed camera and noise jobs are explicit references, never inferred metric authority", { skip: !chromium }, async (t) => {
  const { page, requests, jobs, emit, errors } = await fixture(t, { scans: [{ id: "imu-scan", status: "calibration_ready" }] });
  jobs.push({ id: "camera-complete", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true } }, { id: "camera-failed", mode: "camera", status: "failed" }, { id: "camera-rejected", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: false } }, { id: "noise-complete", mode: "noise", status: "completed", result: { imu_noise_calibrated: false } });
  await emit();
  await openSetup(page);
  await page.locator("#calibration-mode").selectOption("imu");
  await page.locator("#calibration-scan").selectOption("imu-scan");
  await page.locator("#calibration-camera-job").selectOption("camera-complete");
  await page.locator("#calibration-noise-job").selectOption("noise-complete");
  assert.equal(await page.locator('#calibration-camera-job option[value="camera-failed"]').count(), 0);
  assert.equal(await page.locator('#calibration-camera-job option[value="camera-rejected"]').count(), 0);
  await page.locator("#calibration-board-settings").evaluate((element) => { element.open = true; });
  await page.locator("#board-confirmed").check();
  await page.locator('[data-calibration-job="noise-complete"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail")?.textContent.includes('"imu_noise_calibrated": false'));
  await page.locator("#calibration-process").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail")?.textContent.includes("queued"));
  const body = requests.find((request) => request.path === "/api/calibration/jobs" && request.method === "POST").body;
  assert.equal(body.camera_calibration_id, "camera-complete");
  assert.equal(body.noise_calibration_id, "noise-complete");
  assert.equal(Object.hasOwn(body, "metric_vio_allowed"), false);
  assert.deepEqual(errors, []);
});

test("ordinary browser retains upload controls, browser recorder, and shared calibration", { skip: !chromium }, async (t) => {
  const { page, errors } = await fixture(t, { native: false });
  assert.equal(await page.locator("#native-capture-panel").isVisible(), false);
  assert.equal(await page.locator("#record-phone-walk-button").isVisible(), true);
  assert.equal(await page.locator("#existing-video").count(), 1);
  assert.equal(await page.locator("#sensor-bundle").count(), 1);
  await openSetup(page);
  assert.equal(await page.locator("#calibration-capture").isVisible(), false);
  assert.equal(await page.locator("#calibration-form").isVisible(), true);
  assert.deepEqual(errors, []);
});

test("first use exposes capture purpose and Check phone with setup collapsed", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t);
  await page.setViewportSize({ width: 360, height: 640 });
  await emit({ cameras: [], camera_index: -1, room_cameras: [], room_camera_index: -1, calibration_request: {}, status: "Welcome to RoomWalk", detail: "Native phone video and motion, calibration, reconstruction and Noesis alignment in one place.", enabled: { checkPhone: true, configure: true, checkConnection: true } });
  assert.equal(await page.locator("#capture-mode-reconstruction").isVisible(), true);
  assert.equal(await page.locator("#browser-tools").getAttribute("open"), null);
  await page.locator("#native-check-primary").scrollIntoViewIfNeeded();
  const bounds = await page.locator("#native-check-primary").boundingBox();
  assert.ok(bounds.y >= 0 && bounds.y + bounds.height <= 640);
  assert.equal(await page.locator("#native-setup").getAttribute("open"), null);
  assert.match(await page.locator("#native-check-primary").getAttribute("class"), /primary-button/);
  await checkLayout(page, "first-use");
  await openSetup(page);
  assert.equal(await page.locator("#board-confirmed").isVisible(), false);
  assert.equal(await page.locator("#board-squares-x").inputValue(), "10");
  assert.equal(await page.locator("#calibration-board-settings").getAttribute("open"), null);
  assert.match(await page.locator("#calibration-board-summary").textContent(), /10 × 14.*0.018 m \/ 0.0132 m.*300–369/);
  assert.deepEqual(errors, []);
});

test("newly packaged captures open Library with upload beside selection and readable transfer status", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t);
  await emit();
  await emit({ active: true, selected_capture: "take-new" });
  await emit({ active: false, busy: true, selected_capture: "take-new" });
  await emit({ selected_capture: "take-new", artifact: { name: "roomwalk-take-new.zip", bytes: 2_400_000_000 } });
  await page.locator("#panel-library").waitFor({ state: "visible" });
  assert.equal(await page.locator('#native-selected-card [data-native-action="upload"]').isVisible(), true);
  assert.match(await page.locator("#native-artifact").textContent(), /2.4 GB/);
  const transfer = { state: "uploading", active: true, archive_name: "roomwalk-take-new.zip", sent_bytes: 1_200_000_000, total_bytes: 2_400_000_000, bytes_per_second: 10_000_000, elapsed_ms: 120_000 };
  await emit({ selected_capture: "take-new", artifact: { name: "roomwalk-take-new.zip", bytes: 2_400_000_000 }, transfer, enabled: { cancelUpload: true } });
  assert.match(await page.locator("#native-transfer-summary").innerText(), /1.2 GB \/ 2.4 GB/);
  assert.match(await page.locator("#native-transfer-summary").innerText(), /10 MB\/s/);
  assert.match(await page.locator("#native-transfer-summary").innerText(), /Estimated remaining/);
  assert.equal(await page.locator("#native-transfer").isVisible(), false);
  assert.equal(await page.locator('[data-native-action="retryUpload"]').isVisible(), false);
  await checkLayout(page, "transfer-running");
  await emit({ transfer: { ...transfer, state: "failed", active: false, error: "Wi-Fi connection interrupted. The retained ZIP can be retried.", can_retry: true }, enabled: { retryUpload: true } });
  assert.equal(await page.locator('[data-native-action="retryUpload"]').isVisible(), true);
  assert.equal(await page.locator('[data-native-action="cancelUpload"]').isVisible(), false);
  assert.match(await page.locator("#native-transfer-summary").innerText(), /Wi-Fi connection interrupted/);
  await emit({ transfer: { ...transfer, state: "complete", active: false, sent_bytes: transfer.total_bytes, receipt: { validation_status: "pending" } } });
  assert.match(await page.locator("#native-transfer-summary").innerText(), /Storage accepted · import pending/);
  await checkLayout(page, "transfer-stored");
  assert.deepEqual(errors, []);
});

test("camera fit summary and server profile selection are explicit, qualified and new-import-only", { skip: !chromium }, async (t) => {
  const { page, jobs, requests, emit, errors } = await fixture(t);
  jobs.push({ id: "qualified-camera", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true, camera_imu_extrinsics_calibrated: false, time_offset_calibrated: false, imu_noise_calibrated: false, accepted_for_metric_vio: false, quality: { status: "qualified" }, camera_result: { model: "opencv_pinhole", resolution: [7680, 4320], training_rms_px: 0.79, K: [[5031.4, 0, 3839.5], [0, 5030.9, 2159.6], [0, 0, 1]], D: [0.01, -0.03, 0, 0], quality: { heldout: { radial_px: { rms: 0.91, p95: 1.8 } }, coverage_fraction_xy: [0.83, 0.78], tilt_span_deg: 31.2 } } }, artifacts: { camera_profile_url: "/profiles/camera.json", report_url: "/reports/camera.json" } });
  jobs.push({ id: "unqualified-camera", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: false }, artifacts: { camera_profile_url: "/profiles/unqualified.json" } });
  await emit({ cameras: [{ label: "Main rear camera", full_walk_available: true, test_available: true, focus_locked: true }] });
  await openSetup(page);
  await page.locator('[data-calibration-job="qualified-camera"]').click();
  await page.locator('[data-calibration-select="qualified-camera"]').waitFor();
  assert.match(await page.locator("#calibration-job-detail").innerText(), /Camera fit qualified/);
  await page.getByText("Measurements and checks", { exact: true }).click();
  assert.match(await page.locator(".result-gates").innerText(), /Metric VIO acceptance.*Not accepted/s);
  assert.match(await page.locator("#calibration-job-detail").innerText(), /0.91 px/);
  assert.equal(await page.locator("#calibration-job-detail pre").isVisible(), false);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  await page.locator('[data-calibration-select="qualified-camera"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-camera-selection").textContent.includes("Selected: qualified-camera"));
  assert.deepEqual(requests.find((request) => request.path === "/api/calibration/camera-selection" && request.method === "POST").body, { camera_calibration_id: "qualified-camera" });
  assert.match(await page.locator("#calibration-camera-selection").innerText(), /Existing scans and live Noesis settings are unchanged/);
  await checkLayout(page, "camera-qualified-selected");
  await page.locator("#calibration-clear-camera-selection").click();
  await page.waitForFunction(() => document.querySelector("#calibration-camera-selection").textContent.startsWith("No camera profile selected"));
  assert.deepEqual(requests.filter((request) => request.path === "/api/calibration/camera-selection" && request.method === "POST").at(-1).body, { camera_calibration_id: null });
  await page.locator('[data-calibration-job="unqualified-camera"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail pre").textContent.includes('"id": "unqualified-camera"'));
  assert.equal(await page.locator("[data-calibration-select]").count(), 0);
  assert.deepEqual(errors, []);
});

test("noise outcome shows human gates and safe Allan chart, with raw JSON collapsed", { skip: !chromium }, async (t) => {
  const { page, jobs, emit, errors } = await fixture(t);
  jobs.push({ id: "noise-review", mode: "noise", status: "completed", result: { imu_noise_calibrated: false, accepted_for_metric_vio: false, quality: { status: "insufficient_evidence" }, reason_codes: ["gyroscope_x_random_walk_not_observable"], streams: { gyroscope: { duration_s: 10800, sample_count: 2160000, rate_hz: 200, maximum_gap_s: 0.006, axes: [{ axis: "x", fits: { white: { coefficient: 0.0012, heldout_ratio: 1.1, heldout_passed: true }, random_walk: null } }] } }, limitations: ["Three hours is an initial protocol, not a guarantee that random walk is observable."] }, artifacts: { noise_allan_url: "/api/calibration/jobs/noise-review/files/noise_allan.svg", noise_result_url: "/reports/noise.json" } });
  await emit();
  await openSetup(page);
  await page.locator('[data-calibration-job="noise-review"]').click();
  await page.getByText("Charts, source and logs", { exact: true }).click();
  await page.getByText("Measurements and checks", { exact: true }).click();
  await page.locator(".result-chart").waitFor();
  await page.locator(".result-chart").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => document.querySelector(".result-chart img")?.naturalWidth > 0);
  assert.match(await page.locator("#calibration-job-detail").innerText(), /more evidence needed/);
  assert.match(await page.locator("#calibration-job-detail").innerText(), /Gyroscope x random walk not observable/);
  assert.equal(await page.locator("#calibration-job-detail pre").isVisible(), false);
  await checkLayout(page, "noise-review-chart");
  assert.deepEqual(errors, []);
});

test("guided stationary workflow stays on Calibration and presents only the next action", { skip: !chromium }, async (t) => {
  const { page, requests, errors, emit } = await fixture(t, { expandDetails: false });
  await emit();
  await openSetup(page);
  for (const id of ["calibration-settings", "calibration-profile-details", "calibration-history", "calibration-step-picker"]) assert.equal(await page.locator(`#${id}`).getAttribute("open"), null);
  await page.locator("#calibration-guide-next").click();
  await page.locator("#calibration-guide-action").click();
  const telemetry = { state: "preparing", countdown_seconds: 5, elapsed_seconds: 0, expected_duration_s: 60, accel_samples: 0, gyro_samples: 0 };
  await emit({ imu_preparing: true, busy: true, imu_telemetry: telemetry, enabled: { stopImu: true } });
  assert.match(await page.locator("#calibration-run").innerText(), /Starting in 5s/);
  let bounds = await page.locator("#calibration-run").boundingBox();
  assert.ok(bounds.y >= 0 && bounds.y + bounds.height <= 915, "Countdown and telemetry fit without scrolling");
  await emit({ imu_active: true, last_scan_id: "old-upload", selected_capture: "imu-short", imu_telemetry: { ...telemetry, state: "recording", elapsed_seconds: 12, accel_samples: 2400, gyro_samples: 2398 }, enabled: { stopImu: true } });
  assert.equal(requests.some((r) => r.path === "/api/scans/old-upload"), false, "An old receipt cannot interrupt a new recording");
  assert.match(await page.locator("#calibration-run").innerText(), /12 \/ 60s/);
  assert.match(await page.locator("#calibration-run").innerText(), /2,400 samples/);
  const saved = { selected_capture: "imu-short", artifact: { name: "roomwalk-imu-short.zip", bytes: 1234 }, imu_telemetry: { ...telemetry, state: "complete", elapsed_seconds: 60, accel_samples: 12000, gyro_samples: 12000 } };
  await emit(saved);
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Upload recording");
  assert.equal(await page.locator("#calibration-guide-next").isVisible(), false);
  assert.match(await page.locator("#calibration-guide-content").innerText(), /Recording saved/);
  await page.locator("#calibration-guide-action").click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).action, "upload");
  await emit({ ...saved, transfer: { state: "complete", archive_name: saved.artifact.name, receipt: { capture_id: "wrong-take" } } });
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Uploading…", "Unrelated receipt cannot advance this capture");
  await emit({ ...saved, transfer: { state: "complete", archive_name: saved.artifact.name, receipt: { capture_id: "imu-short", status: "stored" } } });
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Process recording");
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Next: Camera");
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  assert.equal(await page.locator("[data-calibration-measurements]").getAttribute("open"), null);
  assert.equal(requests.filter((r) => r.path === "/api/calibration/noise-jobs" && r.method === "POST").length, 1);
  await page.locator("#calibration-guide-action").click();
  assert.equal(await page.locator("#calibration-noise-id").inputValue(), "noise-job");
  assert.match(await page.locator("#calibration-guide-content").innerText(), /BOARD FIXED/);
  await checkLayout(page, "guided-workflow");
  assert.deepEqual(errors, []);
});

test("guided rejection shows the required action without exposing all diagnostic measurements", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t, { expandDetails: false, noiseResult: { noise_model_usable: false, quality: { status: "insufficient_evidence" }, reason_codes: ["gyroscope_x_white_heldout_unstable", "gyroscope_y_white_heldout_unstable"] } });
  await emit();
  await openSetup(page);
  await emit({ imu_active: true, selected_capture: "imu-short" });
  const saved = { selected_capture: "imu-short", artifact: { name: "sensors.zip", bytes: 1234 }, imu_telemetry: { state: "complete", elapsed_seconds: 60, expected_duration_s: 60 } };
  await emit(saved);
  await page.locator("#calibration-guide-action").click();
  await emit({ ...saved, transfer: { state: "complete", archive_name: "sensors.zip", receipt: { capture_id: "imu-short", status: "stored" } } });
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Record again");
  assert.match(await page.locator("#calibration-job-detail").innerText(), /Set the phone down during the countdown/);
  assert.equal(await page.locator(".result-gates").first().isVisible(), false);
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  assert.deepEqual(errors, []);
});

test("camera timing test returns to recording, then a full take uploads and processes in place", { skip: !chromium }, async (t) => {
  const scans = [];
  const { page, emit, jobs, requests, errors } = await fixture(t, { expandDetails: false, scans });
  const intent = { schema: "roomwalk.calibration_request.v1", mode: "camera", board: DEFAULT_BOARD, board_geometry_confirmed: true };
  await emit();
  await openSetup(page);
  await emit({ active: true, selected_capture: "timing-test", calibration_request: intent, capture_short_test: true });
  await emit({ selected_capture: "timing-test", calibration_request: intent, capture_short_test: true, artifact: { name: "test.zip", bytes: 1000 } });
  assert.match(await page.locator("#calibration-message").innerText(), /Timing test passed/);
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Open camera coverage capture");
  await emit({ active: true, selected_capture: "camera-take", calibration_request: intent, capture_short_test: false });
  await emit({ selected_capture: "camera-take", calibration_request: intent, artifact: null });
  await emit({ selected_capture: "camera-take", calibration_request: intent, busy: true });
  const saved = { selected_capture: "camera-take", calibration_request: intent, capture_short_test: false, artifact: { name: "camera.zip", bytes: 5000 } };
  await emit(saved);
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Upload recording");
  await page.locator("#calibration-guide-action").click();
  scans.push({ id: "camera-scan", status: "calibration_ready", capture: { calibration_request: intent } });
  await emit({ ...saved, last_scan_id: "camera-scan", transfer: { state: "complete", archive_name: "camera.zip", receipt: { id: "camera-scan", upload_receipt: { capture_id: "camera-take", status: "stored" } } } });
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Process recording" && !document.querySelector("#calibration-guide-action").disabled);
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Processing…");
  jobs[0].status = "completed"; jobs[0].result = { camera_intrinsics_calibrated: true };
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Next: Motion");
  assert.equal(requests.filter((r) => r.method === "POST" && r.path === "/api/calibration/jobs").length, 1);
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  assert.deepEqual(errors, []);
});

test("a stopped-early camera take explains the native failure instead of claiming timing or calibration completion", { skip: !chromium }, async (t) => {
  const { page, emit, requests, errors } = await fixture(t, { expandDetails: false });
  await emit();
  await openSetup(page);
  const intent = { schema: "roomwalk.calibration_request.v1", mode: "camera", board: DEFAULT_BOARD, board_geometry_confirmed: true };
  for (const shortTest of [true, false]) {
    const id = shortTest ? "failed-timing" : "failed-camera";
    await emit({ active: true, selected_capture: id, calibration_request: intent, capture_short_test: shortTest });
    await emit({ selected_capture: id, calibration_request: intent, capture_short_test: shortTest,
      capture_outcome: { capture_id: id, partial: true, stop_reason: "manual_focus_not_confirmed", message: "Recording stopped after 26.9 seconds. Original retained." } });
    assert.match(await page.locator("#calibration-message").innerText(), /stopped after 26.9 seconds/);
    await emit({ selected_capture: id, calibration_request: intent, capture_short_test: shortTest, artifact: { name: `${id}.zip`, bytes: 1200 },
      cameras: [{ label: "Rear main", full_walk_available: true, test_available: true }],
      capture_outcome: { capture_id: id, partial: true, duration_seconds: 26.9, stop_reason: "manual_focus_not_confirmed", message: "Recording stopped after 26.9 seconds. The phone stopped confirming the locked lens/focus. Original retained." } });
    assert.match(await page.locator("#calibration-message").innerText(), /stopped after 26.9 seconds/);
    assert.match(await page.locator("#calibration-guide-content").innerText(), /not a completed calibration take/);
    assert.equal(await page.locator("#calibration-guide-action").innerText(), "Reopen camera to check lens");
    assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  }
  await checkLayout(page, "camera-stopped-early");
  assert.equal(requests.some((r) => r.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("Check reuse explains missing sources and resumes an existing phone Camera take in place", { skip: !chromium }, async (t) => {
  const scans = [];
  const { page, emit, jobs, requests, errors } = await fixture(t, { expandDetails: false, scans });
  jobs.push({ id: "noise-ready", mode: "noise", status: "completed", result: { noise_model_usable: true } });
  const saved = [
    { id: "camera-saved", kind: "calibration", calibration_mode: "camera", short_test: false, upload_ready: true },
    { id: "camera-test", kind: "calibration", calibration_mode: "camera", short_test: true, upload_ready: true },
    { id: "motion-saved", kind: "calibration", calibration_mode: "imu", short_test: false, upload_ready: true },
  ];
  await emit({ saved });
  await openSetup(page);
  await page.locator("#calibration-step-picker summary").click();
  await page.locator('[data-calibration-step="profile"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Continue camera setup");
  assert.match(await page.locator("#calibration-prerequisites").innerText(), /Camera · No processed result yet/);
  assert.match(await page.locator("#calibration-prerequisites").innerText(), /Stationary sensors · 1 processed result/);
  assert.equal(await page.locator("#motion-profile-form").isVisible(), false, "Missing jobs do not leave an empty disabled form");
  assert.equal(await page.locator("#motion-noise-job option").count(), 2);
  await checkLayout(page, "reuse-missing-sources");
  assert.equal(await page.locator("#calibration-guide-next").isVisible(), false);
  await page.locator("#calibration-guide-action").click();
  assert.equal(await page.locator("[data-recovery-phone]").count(), 1, "Only full Camera takes belong in Camera recovery");
  await checkLayout(page, "reuse-existing-camera");
  assert.equal(requests.some((r) => r.method === "POST"), false);
  await page.locator('[data-recovery-phone="camera-saved"]').click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "selectSaved", args: { id: "camera-saved" } });
  const intent = { schema: "roomwalk.calibration_request.v1", capture_id: "camera-saved", mode: "camera", short_test: false, board: DEFAULT_BOARD, board_geometry_confirmed: true };
  const selected = { saved, selected_capture: "camera-saved", calibration_request: intent, artifact: { name: "camera-saved.zip", bytes: 10000 } };
  await emit(selected);
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Upload recording");
  assert.equal(await page.locator("#tab-setup").getAttribute("aria-selected"), "true");
  assert.equal(requests.some((r) => r.method === "POST"), false);
  await page.locator("#calibration-guide-action").click();
  scans.push({ id: "restored-camera", status: "calibration_ready", capture: { capture_id: "camera-saved", calibration_request: intent } });
  await emit({ ...selected, last_scan_id: "restored-camera", transfer: { state: "complete", archive_name: "camera-saved.zip", receipt: { id: "restored-camera", upload_receipt: { capture_id: "camera-saved" } } } });
  await page.waitForFunction(() => !document.querySelector("#calibration-guide-action").disabled && document.querySelector("#calibration-guide-action").textContent === "Process recording");
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Processing…");
  assert.equal(requests.find((r) => r.method === "POST" && r.path === "/api/calibration/jobs").body.scan_id, "restored-camera");
  assert.deepEqual(errors, []);
});

test("uploaded Camera recovery resumes the exact scan without another upload", { skip: !chromium }, async (t) => {
  const intent = { schema: "roomwalk.calibration_request.v1", capture_id: "retained-camera", mode: "camera", short_test: false, board: DEFAULT_BOARD, board_geometry_confirmed: true };
  const scans = [
    { id: "uploaded-camera", status: "calibration_ready", capture: { capture_id: "retained-camera", calibration_request: intent } },
    { id: "uploaded-test", status: "calibration_ready", capture: { calibration_request: { ...intent, short_test: true } } },
  ];
  const { page, emit, jobs, requests, errors } = await fixture(t, { expandDetails: false, scans });
  jobs.push({ id: "noise-ready", mode: "noise", status: "completed", result: { noise_model_usable: true } });
  await emit();
  await openSetup(page);
  await page.locator("#calibration-step-picker summary").click();
  await page.locator('[data-calibration-step="profile"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Continue camera setup");
  await page.locator("#calibration-guide-action").click();
  assert.equal(await page.locator("[data-recovery-server]").count(), 1);
  await page.locator('[data-recovery-server="uploaded-camera"]').click();
  assert.equal(await page.locator("#calibration-guide-action").innerText(), "Process recording");
  assert.equal(requests.some((r) => r.method === "POST"), false);
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Processing…");
  assert.equal(requests.find((r) => r.method === "POST" && r.path === "/api/calibration/jobs").body.scan_id, "uploaded-camera");
  assert.equal((await page.evaluate(() => window.__nativeCommands)).some((command) => command.action === "upload"), false);
  assert.deepEqual(errors, []);
});

test("phone recovery refuses a calibration sidecar belonging to another capture", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, requests, errors } = await fixture(t, { expandDetails: false });
  jobs.push({ id: "noise-ready", mode: "noise", status: "completed", result: { noise_model_usable: true } });
  const saved = [{ id: "camera-saved", calibration_mode: "camera", upload_ready: true }];
  await emit({ saved });
  await openSetup(page);
  await page.locator("#calibration-step-picker summary").click();
  await page.locator('[data-calibration-step="profile"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Continue camera setup");
  await page.locator("#calibration-guide-action").click();
  await page.locator('[data-recovery-phone="camera-saved"]').click();
  await emit({ saved, selected_capture: "camera-saved", artifact: { name: "camera-saved.zip", bytes: 1000 }, calibration_request: { schema: "roomwalk.calibration_request.v1", capture_id: "wrong-capture", mode: "camera", board: DEFAULT_BOARD, board_geometry_confirmed: true } });
  assert.match(await page.locator("#calibration-message").innerText(), /settings could not be restored/);
  assert.equal(requests.some((r) => r.method === "POST"), false);
  assert.equal((await page.evaluate(() => window.__nativeCommands)).some((command) => command.action === "upload"), false);
  assert.deepEqual(errors, []);
});

test("profile results loading failure is visible and retry refreshes source dropdowns", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, requests, errors } = await fixture(t, { expandDetails: false });
  jobs.push({ id: "noise-ready", mode: "noise", status: "completed", result: { noise_model_usable: true } });
  await page.route("**/api/calibration/jobs", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Temporary server outage" }) }));
  await emit();
  await openSetup(page);
  await page.locator("#calibration-step-picker summary").click();
  await page.locator('[data-calibration-step="profile"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Refresh calibration results");
  assert.match(await page.locator("#calibration-prerequisites").innerText(), /Could not load calibration results/);
  await page.unroute("**/api/calibration/jobs");
  await page.locator("#calibration-guide-action").click();
  await page.waitForFunction(() => document.querySelector("#calibration-guide-action").textContent === "Continue camera setup");
  assert.equal(await page.locator("#motion-noise-job option").count(), 2);
  assert.equal(requests.some((r) => r.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("walk completion opens its saved ZIP despite an intermediate idle camera-close snapshot", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t, { expandDetails: false });
  await emit();
  await openSetup(page);
  await page.locator("#calibration-step-picker summary").click();
  await page.locator('[data-calibration-step="walk"]').click();
  assert.match(await page.locator("#calibration-guide-content").innerText(), /no motion profile selected/);
  await page.locator("#calibration-guide-action").click();
  await emit({ selected_capture: "walk-54s", active: true });
  await emit({ selected_capture: "walk-54s", busy: false });
  await emit({ selected_capture: "walk-54s", busy: true });
  await emit({ selected_capture: "walk-54s", artifact: { name: "walk-54s.zip", bytes: 863000000 } });
  assert.equal(await page.locator("#tab-library").getAttribute("aria-selected"), "true");
  assert.match(await page.locator("#native-selected-card").innerText(), /walk-54s.zip/);
  assert.equal(await page.getByRole("button", { name: "Upload selected capture", exact: true }).isEnabled(), true);
  assert.deepEqual(errors, []);
});

test("selected scan shows verified profile mismatch, while health capability reporting never blocks local capture", { skip: !chromium }, async (t) => {
  const scans = [{ id: "profile-mismatch", name: "Matching profile check", status: "ready", camera_calibration_selection: { status: "not_applied", profile_id: "camera-profile-1", reason_codes: ["native_locked_focus_mismatch"], accepted_for_metric_vio: false }, capture: { camera_calibration_selection: { profile_id: "camera-profile-1" } } }];
  const { page, emit, errors } = await fixture(t, { scans, health: { version: "1.10.0", calibration: { schema: "roomwalk.calibration_capabilities.v1", camera_processing: true, stationary_noise_processing: true, camera_imu_solver_configured: false, camera_selection_scope: "future_exactly_matching_native_imports", automatic_metric_vio_admission: false } } });
  await emit();
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="profile-mismatch"]').click();
  assert.match(await page.locator("[data-scan-camera-selection]").innerText(), /Selected camera profile not applied/);
  assert.match(await page.locator("[data-scan-camera-selection]").innerText(), /native_locked_focus_mismatch/);
  await checkLayout(page, "scan-profile-mismatch");
  await openSetup(page);
  await page.locator("#calibration-mode").selectOption("imu");
  assert.match(await page.locator("#calibration-capabilities").innerText(), /camera–IMU solver not configured/);
  assert.equal(await page.locator("#calibration-capture").isEnabled(), true);
  assert.equal(await page.locator("#calibration-test").isEnabled(), true);
  assert.deepEqual(errors, []);
});

test("motion guide offers the missing phone check after app reopen without recording or losing references", { skip: !chromium }, async (t) => {
  const { page, emit, errors } = await fixture(t);
  await emit({ cameras: [], camera_index: -1, enabled: { checkPhone: true, configure: true, capture: false } });
  await openSetup(page);
  await page.evaluate(() => {
    document.querySelector("#calibration-camera-id").value = "qualified-camera";
    document.querySelector("#calibration-noise-id").value = "saved-noise";
    document.querySelector("#board-confirmed").checked = true;
  });
  await page.locator('[data-calibration-step="imu"]').click();
  const action = page.locator("#calibration-guide-action");
  assert.equal(await action.isEnabled(), true);
  assert.match(await action.innerText(), /Check phone/);
  assert.match(await page.locator("#calibration-capture-readiness").innerText(), /not.*recording/i);
  await action.click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).action, "checkPhone");
  assert.equal(await page.locator("#calibration-camera-id").inputValue(), "qualified-camera");
  await emit();
  assert.equal(await action.innerText(), "Open dynamic phone capture");
  assert.equal(await action.isEnabled(), true);
  await action.click();
  const command = (await page.evaluate(() => window.__nativeCommands)).at(-1);
  assert.equal(command.action, "capture");
  assert.equal(command.args.mode, "imu");
  assert.equal(command.args.camera_calibration_id, "qualified-camera");
  assert.equal(command.args.noise_calibration_id, "saved-noise");
  assert.deepEqual(errors, []);
});

test("motion setup blocks missing references before capture or processing and persists explicit board confirmation", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, requests, errors } = await fixture(t, { scans: [{ id: "retained-motion", status: "calibration_ready" }] });
  jobs.push({ id: "qualified-camera", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true } });
  await emit();
  await openSetup(page);
  await page.locator('[data-calibration-step="imu"]').click();
  await page.locator("#calibration-scan").selectOption("retained-motion");
  await page.locator("#calibration-process").click();
  assert.match(await page.locator("#calibration-message").innerText(), /Choose a qualified camera/);
  assert.equal(requests.some((row) => row.method === "POST"), false);
  const action = page.locator("#calibration-guide-action");
  assert.equal(await action.innerText(), "Review calibration settings");
  await action.click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).some((command) => command.action === "capture"), false);
  await page.locator("#calibration-camera-job").selectOption("qualified-camera");
  assert.match(await page.locator("#calibration-capture-readiness").innerText(), /measured printed board/);
  await action.click();
  await page.locator("#board-confirmed").check();
  assert.equal(await action.innerText(), "Open dynamic phone capture");
  await action.click();
  const capture = (await page.evaluate(() => window.__nativeCommands)).at(-1);
  assert.equal(capture.action, "capture");
  assert.equal(capture.args.camera_calibration_id, "qualified-camera");
  assert.equal(capture.args.board_geometry_confirmed, true);
  await page.reload();
  await page.waitForFunction(() => Boolean(document.querySelector("#board-confirmed")));
  assert.equal(await page.locator("#board-confirmed").isChecked(), true);
  assert.deepEqual(errors, []);
});

test("a temporary polling failure visibly marks stale progress and automatically recovers terminal status", { skip: !chromium }, async (t) => {
  const { page, jobs, requests } = await fixture(t);
  jobs.push({ id: "running-motion", mode: "imu", status: "running", progress: 0.21, message: "Detecting target" });
  let calls = 0;
  await page.route("**/api/calibration/jobs", async (route) => {
    calls += 1;
    if (calls === 2) return route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Temporary connection failure" }) });
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({ jobs }) });
  });
  await openSetup(page);
  await page.locator('[data-calibration-job="running-motion"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail").textContent.includes("21%"));
  await page.waitForFunction(() => document.querySelector("#calibration-message").textContent.includes("may be stale"));
  Object.assign(jobs[0], { status: "failed", progress: 1, result: { error: "Missing camera reference", reason_codes: ["calibration_processing_failed"] } });
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail").textContent.includes("Missing camera reference"), null, { timeout: 15000 });
  assert.ok(calls >= 3);
  assert.equal(requests.some((row) => row.method === "POST"), false);
});

test("the loaded camera reference blocks the observed phone focus mismatch before recording", { skip: !chromium }, async (t) => {
  const { page, emit, jobs } = await fixture(t);
  const focus = { mode: "manual_locked", physical_camera_id: "5", build_fingerprint: "phone-os", focus_distance_diopters: 3.4364262 };
  jobs.push({ id: "camera-reference", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true, camera_result: { binding: { signature: { focus_build_fingerprint: "phone-os", actual_camera2: { lens_focus_distance_diopters: 2.631579, active_physical_camera_id: "5" } } } } } });
  await emit({ cameras: [{ id: "0", focus_locked: true, focus_control: focus }] });
  await openSetup(page);
  await page.locator('[data-calibration-step="imu"]').click();
  await page.locator("#calibration-camera-job").selectOption("camera-reference");
  await page.locator("#calibration-board-settings").evaluate((element) => { element.open = true; });
  await page.locator("#board-confirmed").check();
  assert.match(await page.locator("#calibration-capture-readiness").innerText(), /does not match.*2.631579/);
  await page.locator("#calibration-guide-action").click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).some((command) => command.action === "capture"), false);
  const matching = { ...focus, focus_distance_diopters: Math.fround(2.631579) };
  await emit({ cameras: [{ id: "0", focus_locked: true, focus_control: matching }] });
  await page.locator("#calibration-guide-action").click();
  const command = (await page.evaluate(() => window.__nativeCommands)).at(-1);
  assert.equal(command.action, "capture");
  assert.deepEqual(command.args.calibration_focus_guard, matching);
});

test("guide advances instructions only and keeps native guidance independent from quality", { skip: !chromium }, async (t) => {
  const { page, emit, requests, errors } = await fixture(t);
  await emit();
  await openSetup(page);
  assert.match(await page.locator("#calibration-guide-content").innerText(), /Fix the board/);
  await page.locator("#calibration-guide-next").click();
  assert.equal(await page.locator('[data-calibration-step="noise"]').getAttribute("aria-current"), "step");
  await page.locator("#calibration-guide-action").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "startImu", args: { minutes: 1 } });
  assert.equal(await page.locator("#calibration-guide-action").isDisabled(), true);
  await emit({ imu_active: true, imu_progress: "Stationary sensors · 12s", enabled: { stopImu: true } });
  assert.match(await page.locator("#calibration-guide-progress").innerText(), /Stationary sensors · 12s/);
  await emit({ cameras: [{ focus_locked: true, full_walk_available: true }] });
  await page.locator("#calibration-guide-next").click();
  assert.match(await page.locator("#calibration-guide-content").innerText(), /BOARD FIXED/);
  await page.locator("#calibration-guide-action").click();
  const camera = (await page.evaluate(() => window.__nativeCommands)).at(-1);
  assert.equal(camera.args.mode, "camera");
  assert.equal(Object.hasOwn(camera.args, "duration_seconds"), false);
  await emit();
  await page.locator("#calibration-guide-next").click();
  assert.equal(await page.locator("#board-confirmed").isChecked(), false);
  await page.evaluate(() => {
    document.querySelector("#calibration-camera-id").value = "qualified-camera";
    const confirmed = document.querySelector("#board-confirmed");
    confirmed.checked = true;
    confirmed.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.locator("#calibration-guide-action").click();
  assert.equal((await page.evaluate(() => window.__nativeCommands)).at(-1).args.mode, "imu");
  await emit({ active: true, capture_guidance: { mode: "imu", phase: "translation", title: "Move the PHONE", instruction: "Board stays FIXED", elapsed_seconds: 45, target_seconds: 90, progress: 0.5, stationary: false, completed: false } });
  assert.match(await page.locator("#calibration-guide-progress").innerText(), /45s elapsed · 1m 30s target/);
  assert.equal(await page.locator("#calibration-guide-progress progress").getAttribute("value"), "50");
  await emit({ capture_guidance: { mode: "imu", title: "Capture finished", elapsed_seconds: 90, target_seconds: 90, completed: true } });
  assert.match(await page.locator("#calibration-guide-progress").innerText(), /upload and verify/);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  assert.equal(await page.locator("#board-confirmed").isChecked(), true);
  await checkLayout(page, "short-session-guide");
  assert.deepEqual(errors, []);
});

test("motion profile check and selection require explicit source jobs and a completed ready result", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, requests, errors } = await fixture(t);
  const ids = { camera_calibration_id: "camera-source", imu_calibration_id: "imu-source", noise_calibration_id: "noise-source" };
  jobs.push({ id: "camera-source", mode: "camera", status: "completed", result: { camera_intrinsics_calibrated: true } },
    { id: "imu-source", mode: "imu", status: "completed", request: ids, result: { camera_imu_extrinsics_calibrated: true, time_offset_calibrated: true } },
    { id: "noise-source", mode: "noise", status: "completed", result: { noise_model_usable: true, imu_noise_calibrated: false } });
  await emit();
  await openSetup(page);
  assert.equal(await page.locator("#motion-profile-check").isDisabled(), true);
  await page.locator('[data-calibration-job="imu-source"]').click();
  await page.locator('[data-motion-source="imu-source"]').click();
  assert.equal(await page.locator("#motion-camera-job").inputValue(), "camera-source");
  assert.equal(await page.locator("#motion-noise-job").inputValue(), "noise-source");
  assert.equal(requests.some((request) => request.method === "POST"), false);
  await page.locator("#motion-profile-check").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail pre")?.textContent.includes('"id": "motion-check"'));
  assert.deepEqual(requests.find((request) => request.path === "/api/calibration/motion-profile-jobs" && request.method === "POST").body, ids);
  assert.equal(await page.locator("[data-motion-select]").count(), 0);
  const profile = jobs.find((job) => job.id === "motion-check");
  profile.status = "completed";
  await page.locator("#calibration-refresh").click();
  await page.waitForFunction(() => document.querySelector("#calibration-job-detail pre")?.textContent.includes('"status": "completed"'));
  assert.equal(await page.locator("[data-motion-select]").count(), 0);
  profile.result.motion_profile_ready = true;
  await page.locator("#calibration-refresh").click();
  await page.locator('[data-motion-select="motion-check"]').waitFor();
  assert.match(await page.locator("[data-calibration-measurements]").textContent(), /not a five-minute room-accuracy certificate/);
  await page.locator('[data-motion-select="motion-check"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-vio-profile-status")?.textContent.includes("Selected motion profile: motion-check"));
  assert.deepEqual(requests.find((request) => request.path === "/api/calibration/motion-selection" && request.method === "POST").body, { motion_calibration_id: "motion-check" });
  assert.equal(await page.locator('[data-calibration-step="walk"]').getAttribute("aria-current"), "step");
  assert.match(await page.locator("#calibration-vio-profile-status").innerText(), /first normal walk separately/);
  await page.locator("#calibration-guide-action").click();
  assert.deepEqual((await page.evaluate(() => window.__nativeCommands)).at(-1), { action: "capture", args: { mode: "reconstruction" } });
  await checkLayout(page, "motion-profile-ready");
  await page.locator("#motion-selection-clear").click();
  await page.waitForFunction(() => document.querySelector("#calibration-vio-profile-status")?.textContent.startsWith("No motion profile selected"));
  assert.deepEqual(requests.filter((request) => request.path === "/api/calibration/motion-selection" && request.method === "POST").at(-1).body, { motion_calibration_id: null });
  assert.equal(requests.some((request) => /initiate|run-vio|promote/.test(request.path)), false);
  assert.deepEqual(errors, []);
});

test("missing motion backend is actionable and does not block local short capture", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, errors, requests } = await fixture(t, { health: { calibration: { schema: "roomwalk.calibration_capabilities.v1", motion_profile_validation_available: false, motion_profile_unavailable_reason: "OpenVINS executable is missing." } } });
  for (const mode of ["camera", "imu", "noise"]) jobs.push({ id: `${mode}-job`, mode, status: "completed" });
  await emit();
  await openSetup(page);
  for (const mode of ["camera", "imu", "noise"]) await page.locator(`#motion-${mode}-job`).selectOption(`${mode}-job`);
  assert.match(await page.locator("#motion-profile-capability").innerText(), /OpenVINS executable is missing/);
  assert.match(await page.locator("#motion-profile-capability").innerText(), /Recording longer will not fix/);
  assert.equal(await page.locator("#motion-profile-check").isDisabled(), true);
  assert.equal(await page.locator("#calibration-capture").isEnabled(), true);
  assert.equal(await page.locator("#noise-start").isEnabled(), true);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  assert.deepEqual(errors, []);
});

test("unconfirmed motion selection never looks applied and can be refreshed safely", { skip: !chromium }, async (t) => {
  const { page, emit, jobs, errors } = await fixture(t);
  jobs.push({ id: "ready-motion", mode: "motion_profile", status: "completed", result: { motion_profile_ready: true } });
  await emit();
  await openSetup(page);
  await page.locator('[data-calibration-job="ready-motion"]').click();
  await page.route("**/api/calibration/motion-selection", (route) => route.request().method() === "POST"
    ? route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ selection: { motion_calibration_id: "different-job", profile_id: "profile", status: "ready_for_short_walks", maximum_duration_s: 300 } }) })
    : route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ selection: null }) }));
  await page.locator('[data-motion-select="ready-motion"]').click();
  await page.waitForFunction(() => document.querySelector("#calibration-vio-profile-status")?.textContent.includes("unconfirmed"));
  assert.equal(await page.locator("#motion-selection-clear").isDisabled(), true);
  assert.notEqual(await page.locator('[data-calibration-step="walk"]').getAttribute("aria-current"), "step");
  await page.locator("#motion-selection-refresh").click();
  await page.waitForFunction(() => document.querySelector("#calibration-vio-profile-status")?.textContent.startsWith("No motion profile selected"));
  assert.deepEqual(errors, []);
});

test("scan shows applied or rejected motion binding and retries only after an explicit click", { skip: !chromium }, async (t) => {
  const scans = [{ id: "motion-scan", name: "Profile mismatch", status: "ready", prepared: { frames: [], frame_count: 0 }, capture: { capture_kind: "android_camera_imu", motion_calibration_selection: { motion_calibration_id: "profile-job" }, metric_vio_allowed: false, motion_profile: { status: "not_applied", message: "Phone focus differs from the saved profile", next_action: "Restore matching locked focus and record a new walk; retain this RGB recording." } } }];
  const { page, emit, requests, errors } = await fixture(t, { scans });
  await emit();
  await page.getByRole("tab", { name: "Library", exact: true }).click();
  await page.locator('[data-scan-id="motion-scan"]').click();
  assert.match(await page.locator("[data-scan-motion-profile]").innerText(), /Phone focus differs/);
  assert.match(await page.locator("[data-scan-motion-profile]").innerText(), /RGB remains available/);
  assert.equal(await page.locator("#initiate-vio").count(), 1);
  assert.equal(requests.some((request) => request.method === "POST"), false);
  await page.locator("#initiate-vio").click();
  await page.waitForFunction(() => document.querySelector("#scan-detail")?.textContent.includes("Rechecking motion profile"));
  assert.equal(requests.filter((request) => request.path === "/api/scans/motion-scan/initiate-vio" && request.method === "POST").length, 1);
  assert.equal(await page.locator("#initiate-vio").count(), 0);
  assert.deepEqual(errors, []);
});
