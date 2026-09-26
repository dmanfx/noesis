import assert from "node:assert/strict";
import test from "node:test";
import {
  NATIVE_UI_SCHEMA, WALK_INTENT_SCHEMA, NativeBridge, NativeCaptureUI, isRoomWalkAndroid,
  nativeCommandUrl, parseNativeSnapshot, savedCaptureCards, validateNativeConfiguration,
  pathReferenceForScan, pathSelectionLabel, validatePathReferenceSelection,
  readableBytes, readableDuration, transferPresentation, transferCardMarkup,
  captureGuidanceMarkup,
} from "./static/native_capture.js";

const android = () => ({ navigator: { userAgent: "Mozilla/5.0 RoomWalkAndroid/0.2" }, location: { href: "https://roomwalk.test/", protocol: "https:", origin: "https://roomwalk.test" } });
const snapshot = (fields = {}) => ({ schema: NATIVE_UI_SCHEMA, cameras: [], room_cameras: [], saved: [], enabled: {}, ...fields });

test("stationary telemetry uses native countdown, time and counts without inventing elapsed recording", () => {
  const sensors = { state: "preparing", countdown_seconds: 5, elapsed_seconds: 0, expected_duration_s: 60, accel_samples: 0, gyro_samples: 0 };
  assert.match(captureGuidanceMarkup({ imu_preparing: true, imu_telemetry: sensors }), /Starting in 5s/);
  assert.match(captureGuidanceMarkup({ imu_preparing: true, imu_telemetry: sensors }), /Not recording yet/);
  const active = captureGuidanceMarkup({ imu_active: true, imu_telemetry: { ...sensors, state: "recording", elapsed_seconds: 12, accel_samples: 2400, gyro_samples: 2398 } });
  assert.match(active, /12 \/ 60s/);
  assert.match(active, /2,400 samples/);
  assert.match(active, /2,398 samples/);
  assert.doesNotMatch(active, /Starting in/);
});

test("native detection survives a late injected flag but does not trust an android query parameter", () => {
  assert.equal(isRoomWalkAndroid(android()), true);
  assert.equal(isRoomWalkAndroid({ __ROOMWALK_ANDROID__: true }), true);
  assert.equal(isRoomWalkAndroid({ navigator: { userAgent: "Chrome Android" }, location: { search: "?android=1" } }), false);
});

test("optional native guidance shows actual elapsed progress, never page time or qualification", () => {
  const guide = { mode: "imu", phase: "translation", title: "Move the PHONE", instruction: "Board stays FIXED <script>", elapsed_seconds: 45, target_seconds: 90, progress: 0.99, stationary: false, completed: false };
  const markup = captureGuidanceMarkup({ active: true, capture_guidance: guide });
  assert.match(markup, /45s elapsed · 1m 30s target/);
  assert.match(markup, /value="50"/);
  assert.ok(!markup.includes("<script>"));
  assert.match(markup, /not evidence that calibration passed/);
  assert.match(captureGuidanceMarkup({ capture_guidance: { ...guide, elapsed_seconds: 95, completed: true } }), /value="100"/);
  assert.match(captureGuidanceMarkup({ capture_guidance: { ...guide, elapsed_seconds: 95, completed: true } }), /upload and verify/);
  assert.doesNotMatch(captureGuidanceMarkup({ capture_guidance: { ...guide, elapsed_seconds: null } }), /<progress/);
  assert.match(captureGuidanceMarkup({ active: true }), /elapsed capture time is not reported/);
  assert.match(captureGuidanceMarkup({ imu_active: true }), /Keep the phone untouched/);
});

test("path-refinement guidance enforces a self-carried whole-body posture", () => {
  const markup = captureGuidanceMarkup({ capture_guidance: {
    mode: "path_refinement",
    title: "Film another person",
    instruction: "Film another person by sweeping your arm and vary the phone-body distance while walking near their body route.",
    elapsed_seconds: 12,
    target_seconds: 60,
  } });
  assert.match(markup, /Self-carried path refinement/);
  assert.match(markup, /against your own torso/);
  assert.match(markup, /elbows tucked and stable/);
  assert.match(markup, /move your whole body with the phone/);
  assert.match(markup, /Do not film another person/);
  assert.match(markup, /sweep the phone with your arm/);
  assert.match(markup, /vary phone-to-body distance/);
  assert.doesNotMatch(markup, /<strong>Film another person<\/strong>/);
  assert.doesNotMatch(markup, /Film another person by sweeping your arm/);
});

test("bridge URL exactly implements the bounded action/args protocol", () => {
  const args = { server: "https://roomwalk.test:8789", label: 'name & </script> " quoted' };
  const url = new URL(nativeCommandUrl("configure", args));
  assert.equal(url.protocol, "roomwalk-native:");
  assert.equal(url.host, "action");
  assert.deepEqual(JSON.parse(url.searchParams.get("request")), { action: "configure", args });
  assert.throws(() => nativeCommandUrl("readFile", { path: "/private" }), /Unknown native action/);
  assert.throws(() => nativeCommandUrl("configure", []), /must be an object/);
  assert.throws(() => nativeCommandUrl("configure", { data: "x".repeat(33_000) }), /too large/);
});

test("receive is registered before the initial snapshot request and preserves native integer strings", () => {
  const platform = android();
  let received;
  const bridge = new NativeBridge({ platform, onSnapshot: (state) => { received = state; } });
  assert.equal(typeof platform.RoomWalkNative.receive, "function");
  bridge.send("snapshot");
  assert.match(platform.location.href, /^roomwalk-native:\/\/action\?request=/);
  const state = snapshot({ transfer: { time_ns: "9223372036854775807" } });
  assert.equal(platform.RoomWalkNative.receive(JSON.stringify(state)), true);
  assert.equal(received.transfer.time_ns, "9223372036854775807");
});

test("ordinary browser cannot dispatch native actions, and unsupported native messages stay rejected", () => {
  const browser = { location: { href: "https://roomwalk.test/" } };
  const bridge = new NativeBridge({ platform: browser });
  assert.throws(() => bridge.send("capture", { mode: "walk" }), /requires the RoomWalk Android app/);
  assert.equal(bridge.receive(snapshot()), false);
  assert.throws(() => parseNativeSnapshot({ schema: "other" }), /unsupported native status/);
  assert.throws(() => parseNativeSnapshot("not JSON"));
  const failures = [];
  const native = new NativeBridge({ platform: android(), onError: (error) => failures.push(error) });
  assert.equal(native.receive({ schema: "other" }), false);
  assert.equal(failures.length, 1);
});

test("native configuration requires a trusted origin and bounded diagnostic inputs", () => {
  const values = { server: "https://roomwalk.test:8789/", camera_index: "0", room_camera_index: "1", imu_minutes: "1" };
  assert.deepEqual(validateNativeConfiguration(values), { server: "https://roomwalk.test:8789", camera_index: 0, room_camera_index: 1, imu_minutes: 1 });
  for (const server of ["http://roomwalk.test", "https://user:pass@roomwalk.test", "https://roomwalk.test/path", "https://roomwalk.test/?key=secret", "javascript:alert(1)"]) {
    assert.throws(() => validateNativeConfiguration({ ...values, server }), /HTTPS/);
  }
  assert.throws(() => validateNativeConfiguration({ ...values, imu_minutes: 180 }), /1 and 5/);
  assert.throws(() => validateNativeConfiguration({ ...values, camera_index: 1.5 }), /valid camera/);
});

test("saved capture markup escapes names and IDs and fails closed on disabled actions", () => {
  const markup = savedCaptureCards(parseNativeSnapshot(snapshot({ saved: [{ id: 'x" onclick="alert(1)', name: "<img src=x onerror=alert(1)>" }] })));
  assert.ok(markup.includes("&lt;img"));
  assert.ok(markup.includes("&quot;"));
  assert.ok(!markup.includes("<img"));
  assert.match(markup, /disabled/);
});

test("native UI uses enabled actions, forwards exact calibration intent, and notifies only a new receipt identity", () => {
  const platform = android();
  const uploads = [];
  const failures = [];
  const ui = new NativeCaptureUI({ platform, onUploaded: (id) => uploads.push(id), onError: (error) => failures.push(error) });
  assert.equal(ui.capture("camera", {}), false);
  ui.receive(snapshot({ last_scan_id: "previous", enabled: { capture: true, checkConnection: true } }));
  assert.deepEqual(uploads, []);
  const board = { squares_x: 10, marker_ids: [300, 301] };
  assert.equal(ui.capture("imu", board, { board_geometry_confirmed: true, camera_calibration_id: "cam-123", noise_calibration_id: "noise-234" }), true);
  assert.deepEqual(JSON.parse(new URL(platform.location.href).searchParams.get("request")), {
    action: "capture", args: { mode: "imu", board, board_geometry_confirmed: true, camera_calibration_id: "cam-123", noise_calibration_id: "noise-234" },
  });
  ui.receive(snapshot({ last_scan_id: "new-scan" }));
  ui.receive(snapshot({ last_scan_id: "new-scan" }));
  assert.deepEqual(uploads, ["new-scan"]);
  ui.configDirty = true;
  assert.equal(ui.capture("walk"), false);
  assert.match(failures.at(-1).message, /Apply the edited phone setup/);
});

test("native capture sends the shared two-mode walk intent and keeps legacy walk as reconstruction", () => {
  const platform = android();
  const failures = [];
  const ui = new NativeCaptureUI({ platform, onError: (error) => failures.push(error) });
  ui.receive(snapshot({ enabled: { capture: true } }));
  assert.equal(ui.capture("path_refinement", undefined, { target_scan_id: "room-42", carry_protocol: "close_body", accuracy_target_m: 0.1 }), true);
  assert.deepEqual(JSON.parse(new URL(platform.location.href).searchParams.get("request")), {
    action: "capture",
    args: { mode: "path_refinement", target_scan_id: "room-42", carry_protocol: "close_body", accuracy_target_m: 0.1 },
  });
  assert.equal(ui.capture("walk"), false, "single-flight still protects the native bridge");
  assert.equal(failures.length, 1);
  assert.match(savedCaptureCards(parseNativeSnapshot(snapshot({ saved: [{ id: "p", name: "Path", walk_intent: { schema: WALK_INTENT_SCHEMA, mode: "path_refinement", target_scan_id: "room-42" } }] }))), /Path refinement/);
});

test("available retained PCF selection survives UI intent and native bridge exactly, while reconstruction rejects it", () => {
  const selection = {
    kind: "scene_prior_pcf",
    prior_id: "pcf_prior_living_room_v3",
    manifest_sha256: "a".repeat(64),
    camera_id: "living_room",
    frame_binding_sha256: "b".repeat(64),
  };
  assert.equal(pathSelectionLabel(selection), "Noesis PCF · living_room · prior pcf_prior_living_room_v3");
  assert.deepEqual(pathReferenceForScan({ path_reference: { status: "available", label: "Noesis PCF · living_room", selection } }).selection, selection);
  assert.deepEqual(validatePathReferenceSelection(selection), selection);
  assert.throws(() => validatePathReferenceSelection({ ...selection, extra: true }), /unsupported fields/);
  assert.throws(() => validatePathReferenceSelection({ ...selection, manifest_sha256: "A".repeat(64) }), /lowercase/);

  const platform = android();
  const failures = [];
  const ui = new NativeCaptureUI({ platform, onError: (error) => failures.push(error) });
  ui.receive(snapshot({ supports_target_reference: true, enabled: { capture: true } }));
  assert.equal(ui.capture("path_refinement", undefined, { target_scan_id: "room-42", target_reference: selection }), true);
  const request = JSON.parse(new URL(platform.location.href).searchParams.get("request"));
  assert.deepEqual(request.args.target_reference, selection);
  assert.deepEqual(ui.captureIntent.target_reference, selection);

  const reconstructionFailures = [];
  const reconstruction = new NativeCaptureUI({ platform: android(), onError: (error) => reconstructionFailures.push(error) });
  reconstruction.receive(snapshot({ supports_target_reference: true, enabled: { capture: true } }));
  assert.equal(reconstruction.capture("reconstruction", undefined, { target_scan_id: "room-42", target_reference: selection }), false);
  assert.match(reconstructionFailures.at(-1).message, /only for path refinement/);
  assert.equal(failures.length, 0);
});

test("old native companion cannot silently drop an available PCF selection", () => {
  const failures = [];
  const ui = new NativeCaptureUI({ platform: android(), onError: (error) => failures.push(error) });
  ui.receive(snapshot({ enabled: { capture: true } }));
  assert.equal(ui.capture("path_refinement", undefined, {
    target_scan_id: "room-42",
    target_reference: { kind: "scene_prior_pcf", prior_id: "pcf_v1", manifest_sha256: "a".repeat(64), camera_id: "living_room", frame_binding_sha256: "b".repeat(64) },
  }), false);
  assert.match(failures.at(-1).message, /cannot preserve the selected PCF reference/);
});

test("optional calibration actions pass their own mode and preserve the chosen walk intent", () => {
  const modes = [];
  const ui = new NativeCaptureUI({ platform: android(), canCapture: (mode) => { modes.push(mode); return mode !== "path_refinement"; } });
  ui.setCaptureIntent({ mode: "path_refinement", target_scan_id: null });
  const intent = { ...ui.captureIntent };
  for (const mode of ["camera", "imu"]) {
    ui.receive(snapshot({ enabled: { capture: true } }));
    assert.equal(ui.capture(mode), true);
    assert.deepEqual(ui.captureIntent, intent);
  }
  ui.receive(snapshot({ enabled: { startImu: true } }));
  assert.equal(ui.startImu(1), true);
  assert.deepEqual(ui.captureIntent, intent);
  assert.deepEqual(modes, ["camera", "imu", "noise"]);
});

test("long stationary protocol is explicit and does not widen short diagnostic actions", () => {
  const platform = android();
  const failures = [];
  const ui = new NativeCaptureUI({ platform, onError: (error) => failures.push(error) });
  ui.receive(snapshot({ enabled: { startImu: true } }));
  assert.equal(ui.startImu(180), false);
  assert.match(failures.at(-1).message, /1–5/);
  assert.equal(ui.startImu(180, { stationaryNoise: true }), true);
  assert.deepEqual(JSON.parse(new URL(platform.location.href).searchParams.get("request")), { action: "startImu", args: { minutes: 180 } });
  assert.equal(ui.startImu(181, { stationaryNoise: true }), false);
  assert.equal(ui.startImu(0, { stationaryNoise: true }), false);
});

test("native text-valued diagnostic duration acknowledges the exact edited setup", () => {
  const ui = new NativeCaptureUI({ platform: android() });
  ui.configDirty = true;
  ui.pendingConfiguration = { server: "https://roomwalk.test", camera_index: 0, room_camera_index: 1, imu_minutes: 3 };
  ui.receive(snapshot({ ...ui.pendingConfiguration, imu_minutes: "1" }));
  assert.equal(ui.configDirty, true);
  ui.receive(snapshot({ ...ui.pendingConfiguration, imu_minutes: "3" }));
  assert.equal(ui.configDirty, false);
  assert.equal(ui.pendingConfiguration, null);
});

test("transfer presentation shows bounded units, speed, elapsed and estimated remaining time", () => {
  assert.equal(readableBytes(1_200_000_000), "1.2 GB");
  assert.equal(readableBytes(0), "0 B");
  assert.equal(readableBytes(null), "Not reported");
  assert.equal(readableDuration(3661), "1h 1m");
  const view = transferPresentation({ state: "uploading", active: true, sent_bytes: 1_200_000_000, total_bytes: 2_400_000_000, elapsed_ms: 120_000, bytes_per_second: 10_000_000 });
  assert.equal(view.percent, 50);
  assert.deepEqual(Object.fromEntries(view.stats), { Transferred: "1.2 GB / 2.4 GB", "Average speed": "10 MB/s", Elapsed: "2m 0s", "Estimated remaining": "2m 0s" });
  assert.equal(transferPresentation({ sent_bytes: Infinity, total_bytes: -1 }).percent, null);
  assert.equal(transferPresentation({ sent_bytes: 120, total_bytes: 100 }).percent, 100);
});

test("a complete transfer is storage evidence, not implicit import or calibration acceptance", () => {
  const pending = transferPresentation({ state: "complete", receipt: { id: "scan", validation_status: "pending" } });
  assert.equal(pending.title, "Storage accepted · import pending");
  assert.match(transferPresentation({ state: "complete" }).title, /receipt not reported/);
  assert.match(transferPresentation({ state: "complete", imu: true, receipt: { status: "stored" } }).description, /Noise qualification has not been established/);
  const markup = transferCardMarkup({ state: "failed", error: "<img src=x onerror=alert(1)>", archive_name: "<script>bad</script>" });
  assert.ok(!markup.includes("<img"));
  assert.ok(!markup.includes("<script>"));
  assert.match(markup, /Upload needs attention/);
});

test("native commands are single-flight with snapshot recovery and no automatic action retry", () => {
  let now = 0;
  const urls = [];
  const bridge = new NativeBridge({ platform: android(), now: () => now, navigate: (url) => urls.push(url) });
  assert.equal(bridge.send("capture", { mode: "walk" }), true);
  assert.throws(() => bridge.send("startImu", { minutes: 1 }), /Waiting for the phone/);
  assert.equal(bridge.send("snapshot"), false);
  assert.equal(urls.length, 1);
  now = 2001;
  assert.equal(bridge.send("snapshot"), true);
  assert.equal(bridge.receive(snapshot()), true);
  assert.equal(bridge.send("checkPhone"), true);
  assert.deepEqual(urls.map((url) => JSON.parse(new URL(url).searchParams.get("request")).action), ["capture", "snapshot", "checkPhone"]);
});

test("only a newly packaged acquisition triggers the Library handoff, not an old selected ZIP", () => {
  const packaged = [];
  const ui = new NativeCaptureUI({ platform: android(), onPackaged: (state) => packaged.push(state.selected_capture) });
  ui.receive(snapshot({ selected_capture: "old", artifact: { name: "old.zip", bytes: 42 } }));
  assert.deepEqual(packaged, []);
  ui.receive(snapshot({ selected_capture: "new-imu", imu_active: true }));
  ui.receive(snapshot({ selected_capture: "new-imu", busy: true, artifact: { name: "manifest.json", bytes: 30 } }));
  ui.receive(snapshot({ selected_capture: "new-imu", artifact: { name: "new.zip", bytes: 3000 } }));
  ui.receive(snapshot({ selected_capture: "new-imu", artifact: { name: "new.zip", bytes: 3000 } }));
  ui.receive(snapshot({ selected_capture: "old", artifact: { name: "old.zip", bytes: 42 } }));
  assert.deepEqual(packaged, ["new-imu"]);
});

test("camera close cannot consume the save handoff before its exact ZIP arrives", () => {
  const packaged = [];
  const ui = new NativeCaptureUI({ platform: android(), onPackaged: (state) => packaged.push(state.selected_capture) });
  ui.receive(snapshot({ selected_capture: "old", artifact: { name: "old.zip", bytes: 42 } }));
  ui.receive(snapshot({ selected_capture: "walk", active: true, busy: true }));
  ui.receive(snapshot({ selected_capture: "walk", active: false, busy: false }));
  ui.receive(snapshot({ selected_capture: "walk", busy: true }));
  ui.receive(snapshot({ selected_capture: "walk", artifact: { name: "walk.zip", bytes: 3000 } }));
  ui.receive(snapshot({ selected_capture: "walk", artifact: { name: "walk.zip", bytes: 3000 } }));
  assert.deepEqual(packaged, ["walk"]);
});

test("a different selected capture cannot complete a pending recording handoff", () => {
  const packaged = [];
  const ui = new NativeCaptureUI({ platform: android(), onPackaged: (state) => packaged.push(state.selected_capture) });
  ui.receive(snapshot({ selected_capture: "recording", active: true }));
  ui.receive(snapshot({ selected_capture: "recording", busy: false }));
  ui.receive(snapshot({ selected_capture: "unrelated", artifact: { name: "unrelated.zip", bytes: 42 } }));
  assert.deepEqual(packaged, []);
});
